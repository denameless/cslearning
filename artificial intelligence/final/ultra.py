import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import re
import random
from tqdm import tqdm
import time

# --- 1. 配置文件 ---
class Config:
    YIZHUIHE_DIR = "./yizhuihe"
    TONGJI_XLSX = "./zhuihetongji.xlsx"
    EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    MARGIN = 2.0
    EMBEDDING_DIM = 128
    IMAGE_SIZE = (224, 224)
    MODEL_PATH = "siamese_net_bamboo.pth"

# --- 2. 数据加载模块 ---
class BambooSlipDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.image_pairs, self.labels = self._create_pairs()

    def _find_image_path(self, slip_number):
        try:
            core_number = str(slip_number).split('-')[-1]
        except:
            core_number = str(slip_number)
        search_key = core_number.zfill(4)
        for root, _, files in os.walk(self.config.YIZHUIHE_DIR):
            for file in files:
                if re.match(f"^{search_key}(_.*|\\.jpg|\\.png|\\.jpeg)?$", file, re.IGNORECASE):
                    return os.path.join(root, file)
        return None

    def _create_pairs(self):
        print(">> 正在创建图像对...")
        try:
            df = pd.read_excel(self.config.TONGJI_XLSX, sheet_name="上下拼", engine='openpyxl', skiprows=1)
        except Exception as e:
            print(f"!! 读取Excel文件 '{self.config.TONGJI_XLSX}' 或工作表 '上下拼' 时出错: {e}")
            return [], []
    
        positive_pairs, all_slips = [], []
        slip_columns = df.columns[1:6] 
        for _, row in df.iterrows():
            current_group_slips = [str(s) for s in row[slip_columns] if pd.notna(s)]
            if not current_group_slips:
                continue
            all_slips.extend(current_group_slips)
            for i in range(len(current_group_slips) - 1):
                path1 = self._find_image_path(current_group_slips[i])
                path2 = self._find_image_path(current_group_slips[i+1])
                if path1 and path2:
                    positive_pairs.append((path1, path2))
    
        print(f">> 成功创建 {len(positive_pairs)} 个正样本对。")
    
        negative_pairs = []
        num_negative_pairs = len(positive_pairs)
        all_slips_unique = list(set(all_slips))
        positive_set = set(positive_pairs)
        positive_set.update([(p[1], p[0]) for p in positive_pairs])
        
        while len(negative_pairs) < num_negative_pairs and len(all_slips_unique) >= 2:
            s1, s2 = random.sample(all_slips_unique, 2)
            p1, p2 = self._find_image_path(s1), self._find_image_path(s2)
            if p1 and p2 and (p1, p2) not in positive_set:
                negative_pairs.append((p1, p2))
        
        print(f">> 成功创建 {len(negative_pairs)} 个负样本对。")
    
        image_pairs = positive_pairs + negative_pairs
        labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        return image_pairs, labels

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        img1_path, img2_path = self.image_pairs[index]
        label = self.labels[index]
        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except (IOError, FileNotFoundError):
            placeholder = torch.zeros((3, self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1]))
            return placeholder, placeholder, torch.tensor(0, dtype=torch.float32)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# --- 3. 模型定义 ---
class _CustomResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(_CustomResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNet, self).__init__()
        self.sobel_layer = self._create_sobel_layer()
        self.texture_cnn = self._build_cnn_branch(in_channels=3)
        self.contour_cnn = self._build_cnn_branch(in_channels=1)
        feature_dim = 256 * 7 * 7
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 8), nn.ReLU(),
            nn.Linear(feature_dim // 8, feature_dim), nn.Sigmoid())
        self.final_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4), nn.ReLU(),
            nn.Linear(feature_dim // 4, embedding_dim))

    def _create_sobel_layer(self):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        sobel_filter.weight.data.copy_(torch.cat((sobel_x, sobel_y), dim=0))
        for param in sobel_filter.parameters():
            param.requires_grad = False
        return sobel_filter

    def _build_cnn_branch(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _CustomResidualBlock(64, 64),
            _CustomResidualBlock(64, 128, stride=2),
            _CustomResidualBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((7, 7)))

    def forward_one(self, x):
        x_gray = transforms.functional.rgb_to_grayscale(x)
        sobel_out = self.sobel_layer(x_gray)
        x_contour = torch.hypot(sobel_out[:, 0:1, :, :], sobel_out[:, 1:2, :, :])
        texture_features_map = self.texture_cnn(x)
        contour_features_map = self.contour_cnn(x_contour)
        texture_vec = torch.flatten(texture_features_map, 1)
        contour_vec = torch.flatten(contour_features_map, 1)
        w = self.fusion_gate(torch.cat([texture_vec, contour_vec], dim=1))
        fused_vec = w * texture_vec + (1 - w) * contour_vec
        embedding = self.final_fc(fused_vec)
        return embedding

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)

# --- 4. 损失函数 ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# --- 5. 训练脚本 ---
def train():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = BambooSlipDataset(config, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    model = SiameseNet(embedding_dim=config.EMBEDDING_DIM).to(device)
    criterion = ContrastiveLoss(margin=config.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("开始训练...")
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (img1, img2, label) in enumerate(dataloader, 1):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 5 == 0:
                print(f"Epoch [{epoch + 1}/{config.EPOCHS}], Batch [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
        epoch_loss = running_loss / len(dataloader)
        print(f"--- Epoch {epoch + 1} 完成, 平均 Loss: {epoch_loss:.4f} ---")
    
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"模型已保存: {config.MODEL_PATH}")

# --- 6. 主程序入口 (已修改为大规模评估) ---
if __name__ == "__main__":
    config = Config()

    if not os.path.exists(config.MODEL_PATH):
        print(f"模型文件 '{config.MODEL_PATH}' 不存在，开始训练模型...")
        train()
    else:
        print(f"发现已训练的模型 '{config.MODEL_PATH}'。")

    print("\n--- 开始加载模型并准备预测 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNet(embedding_dim=config.EMBEDDING_DIM)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n--- 正在构建真值与路径映射... ---")
    ground_truth_map = {}
    try:
        df = pd.read_excel(config.TONGJI_XLSX, sheet_name="上下拼", engine='openpyxl', skiprows=1)
        slip_columns = df.columns[1:6]
        for _, row in df.iterrows():
            current_group_slips = [str(s) for s in row[slip_columns] if pd.notna(s)]
            for i in range(len(current_group_slips) - 1):
                s1, s2 = current_group_slips[i], current_group_slips[i+1]
                ground_truth_map[s1] = s2
                ground_truth_map[s2] = s1
        print(f">> 成功加载 {len(ground_truth_map) // 2} 对真值配对。")
    except Exception as e:
        print(f"!! 错误: 读取真值Excel文件失败: {e}")
        ground_truth_map = {}

    temp_dataset = BambooSlipDataset(config)
    all_image_paths = sorted(list(set([p for pair in temp_dataset.image_pairs for p in pair])))
    
    path_to_slip_map = {}
    slip_to_path_map = {}
    
    temp_slip_map = {}
    for slip in ground_truth_map.keys():
        core_num = slip.split('-')[-1]
        temp_slip_map[core_num.zfill(4)] = slip

    for path in all_image_paths:
        filename = os.path.basename(path)
        core_id_match = re.match(r'^(\d{4})', filename)
        if core_id_match:
            core_id = core_id_match.group(1)
            if core_id in temp_slip_map:
                excel_slip = temp_slip_map[core_id]
                path_to_slip_map[path] = excel_slip
                slip_to_path_map[excel_slip] = path
    print(f">> 成功为 {len(path_to_slip_map)} 个图片路径建立简号映射。")

    print("\n--- 正在为数据集中所有图片生成特征向量... ---")
    path_list = []
    embedding_list = []
    with torch.no_grad():
        for path in tqdm(all_image_paths, desc="生成Embeddings"):
            if path not in path_to_slip_map:
                continue
            try:
                img = Image.open(path).convert("RGB")
                tensor = eval_transform(img).unsqueeze(0).to(device)
                embedding = model.forward_one(tensor)
                path_list.append(path)
                embedding_list.append(embedding)
            except Exception as e:
                print(f"处理图片 {path} 时出错: {e}")

    if not embedding_list:
        print("!! 错误: 未能生成任何特征向量，无法进行预测。")
    else:
        all_embeddings = torch.cat(embedding_list, dim=0)
        print(f"\n成功为 {len(path_list)} 张有效图片生成了特征向量。")

        # --- 新增: 步骤5 - 执行大规模预测并计算Top-K准确率 ---
        print(f"\n{'=' * 20} 开始大规模评估 {'=' * 20}")
        
        K = 5 # 我们将计算Top-1和Top-5准确率

        top_1_hits = 0
        top_5_hits = 0
        total_valid_queries = 0

        # 从所有已知的正样本对中构建我们的评估查询
        eval_pairs = []
        processed_pairs = set()
        for key, value in ground_truth_map.items():
            if tuple(sorted((key, value))) not in processed_pairs:
                # 确保这两个简都在我们的处理列表中
                if slip_to_path_map.get(key) in path_list and slip_to_path_map.get(value) in path_list:
                    eval_pairs.append((key, value))
                    processed_pairs.add(tuple(sorted((key, value))))
        
        if not eval_pairs:
            print("未能构建任何有效的评估对。请检查数据和路径映射。")
        else:
            for anchor_slip, positive_slip in tqdm(eval_pairs, desc="大规模评估中"):
                total_valid_queries += 1
                
                anchor_path = slip_to_path_map[anchor_slip]
                query_index = path_list.index(anchor_path)
                query_embedding = all_embeddings[query_index]

                distances = F.pairwise_distance(query_embedding, all_embeddings)
                
                # 在排序中排除查询对象自身
                distances[query_index] = float('inf')

                _, top_k_indices = torch.topk(distances, k=K, largest=False)

                # 获取Top-K预测结果的简号列表
                predicted_slips = [path_to_slip_map.get(path_list[idx.item()]) for idx in top_k_indices]

                # 检查Top-1命中
                if predicted_slips and predicted_slips[0] == positive_slip:
                    top_1_hits += 1
                
                # 检查Top-5命中
                if positive_slip in predicted_slips:
                    top_5_hits += 1

            # --- 结果报告 ---
            print(f"\n{'=' * 20} 评估结果报告 {'=' * 20}")
            if total_valid_queries > 0:
                top_1_accuracy = (top_1_hits / total_valid_queries) * 100
                top_5_accuracy = (top_5_hits / total_valid_queries) * 100
                
                print(f"有效查询总数 (来自 {len(eval_pairs)} 个正样本对): {total_valid_queries}")
                print(f"\nTop-1 准确率: {top_1_accuracy:.2f}%")
                print(f"(释义: 在{total_valid_queries}次查询中，有{top_1_hits}次模型的首选推荐是完全正确的。)")
                
                print(f"\nTop-5 准确率: {top_5_accuracy:.2f}%")
                print(f"(释义: 在{total_valid_queries}次查询中，有{top_5_hits}次正确答案出现在了模型推荐的前5名中。)")
            else:
                print("未找到任何有效的查询样本，无法计算准确率。")
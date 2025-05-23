import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from multiprocessing import freeze_support

# --- 类定义和函数定义 ---
def torchvision_loader(path: str):
    img_tensor = read_image(path, mode=ImageReadMode.RGB)
    return img_tensor

class CustomTestDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())
        self.samples = []

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        filenames = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(image_extensions)]

        for fname in filenames:
            matched_class = None
            for class_name_key in self.classes:
                if fname.lower().startswith(class_name_key.lower()):
                    matched_class = class_name_key
                    break
            
            if matched_class:
                label_idx = self.class_to_idx[matched_class]
                self.samples.append((os.path.join(self.root_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image_tensor = read_image(img_path, mode=ImageReadMode.RGB)

        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, label, img_path 

class SimpleCNN(nn.Module):
    def __init__(self, num_classes_model, image_size_model=128):
        super(SimpleCNN, self).__init__()
        self.image_size_model = image_size_model
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_input_features = 64 * (self.image_size_model // 8) * (self.image_size_model // 8)
        self.fc1 = nn.Linear(self.fc_input_features, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes_model)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self.fc_input_features)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    print(f"PyTorch Version: {torch.__version__}")

    data_dir = 'cnn图片'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50 
    image_size = 128
    
    best_model_path = "best_model_on_train_acc.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms, loader=torchvision_loader)

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    print(f"从训练集获取的类别: {class_names}")
    print(f"类别到索引的映射: {class_to_idx}")

    num_classes = len(class_names)

    test_dataset = CustomTestDataset(test_dir, class_to_idx, transform=test_transforms) 

    num_workers_val = 0
    if device.type == 'cuda':
        if os.name == 'posix':
            num_workers_val = 4
        else: 
            num_workers_val = 0 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers_val, pin_memory=True if device.type == 'cuda' else False)
    # For test_loader, CustomTestDataset now returns (image, label, path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers_val, pin_memory=True if device.type == 'cuda' else False)

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    net = SimpleCNN(num_classes_model=num_classes, image_size_model=image_size).to(device)
    print("神经网络架构:")
    # print(net) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    epoch_times = []
    
    best_train_accuracy_this_run = 0.0

    print("\n开始训练和评估...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        net.train()
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0

        # train_loader from ImageFolder yields (inputs, labels)
        for i, (inputs, labels) in enumerate(train_loader): 
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss_train / total_train if total_train > 0 else 0
        epoch_train_acc = 100.0 * correct_train / total_train if total_train > 0 else 0
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        epoch_test_loss = float('nan')
        epoch_test_acc = float('nan')
        
        if len(test_dataset) > 0:
            net.eval()
            running_loss_test = 0.0
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for inputs, labels, _ in test_loader: # MODIFIED: Unpack and ignore path
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    running_loss_test += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
            
            epoch_test_loss = running_loss_test / total_test if total_test > 0 else 0
            epoch_test_acc = 100.0 * correct_test / total_test if total_test > 0 else 0
        
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, | '
              f'Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%, '
              f'Time: {epoch_duration:.2f}s')

        if not np.isnan(epoch_train_acc) and epoch_train_acc > best_train_accuracy_this_run:
            best_train_accuracy_this_run = epoch_train_acc
            torch.save(net.state_dict(), best_model_path)
            print(f"  ** 新的最佳训练准确率: {best_train_accuracy_this_run:.2f}%. 模型已保存到 {best_model_path} **")

    total_training_time = time.time() - start_time
    print(f"训练和评估完成！总耗时: {total_training_time:.2f}s")
    
    if num_epochs > 0 and len(train_dataset) > 0:
        print(f"本次训练中达到的最佳训练准确率为: {best_train_accuracy_this_run:.2f}%")

    print(f"\n正在从 {best_model_path} 加载本次训练中的最优模型（基于训练准确率）进行最终评估...")
    final_net = SimpleCNN(num_classes_model=num_classes, image_size_model=image_size).to(device)
    
    model_loaded_for_final_eval = False
    model_source_info = "未知来源或无模型" # For descriptive printing later

    if os.path.exists(best_model_path):
        final_net.load_state_dict(torch.load(best_model_path, map_location=device))
        print("最优模型 (基于训练准确率) 加载成功。")
        model_loaded_for_final_eval = True
        model_source_info = "本次训练最佳训练准确率模型"
    elif num_epochs > 0 and len(train_dataset) > 0 : 
        print(f"未找到最优模型文件 {best_model_path}。将使用最后一个 epoch 的模型状态进行最终评估。")
        final_net.load_state_dict(net.state_dict()) 
        model_loaded_for_final_eval = True
        model_source_info = "最后一个epoch的模型"

    if len(test_dataset) > 0 and model_loaded_for_final_eval:
        final_net.eval() 
        correct_final = 0
        total_final = 0
        with torch.no_grad():
            # test_loader from CustomTestDataset now yields (images, labels, paths)
            # We only need images and labels for accuracy calculation here.
            for images, labels, _ in test_loader: # MODIFIED: Unpack and ignore path
                images, labels = images.to(device), labels.to(device)
                outputs = final_net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_final += labels.size(0)
                correct_final += (predicted == labels).sum().item()
        
        final_test_accuracy = 100.0 * correct_final / total_final if total_final > 0 else 0.0
        print(f'\n（使用{model_source_info}）最终在 {total_final} 个测试样本上的准确率: {final_test_accuracy:.2f} %')
    elif len(test_dataset) == 0:
        print("\n测试集为空，跳过最终的测试评估。")
    elif not model_loaded_for_final_eval:
        print("\n由于没有模型可供评估，跳过最终的测试评估。")


    # --- 单张图片预测结果 ---
    if len(test_dataset) > 0 and model_loaded_for_final_eval:
        print(f"\n--- 单张图片预测结果 (使用 {model_source_info}) ---")
        final_net.eval()
        idx_to_class = {v: k for k, v in class_to_idx.items()} # Create reverse mapping
        
        with torch.no_grad():
            # test_loader from CustomTestDataset yields (images, true_labels, paths)
            for images, true_labels, paths in test_loader: 
                images = images.to(device)
                # true_labels are already on CPU or on device if pin_memory=True and num_workers>0
                # For safety, ensure true_labels used for indexing are on CPU if needed, or items are extracted.
                
                outputs = final_net(images)
                _, predicted_indices = torch.max(outputs.data, 1)

                for i in range(len(paths)): # Iterate through items in the batch
                    img_file_path = paths[i]
                    predicted_idx = predicted_indices[i].item()
                    true_label_idx = true_labels[i].item() # Get true label for this specific image

                    predicted_class_name = idx_to_class[predicted_idx]
                    true_class_name = idx_to_class[true_label_idx]
                    
                    print(f"图片: {os.path.basename(img_file_path)} - 预测类别: {predicted_class_name} (真实类别: {true_class_name})")
    
    elif len(test_dataset) == 0:
        # This condition is already handled for the accuracy part,
        # but good to have a specific message if only this part is skipped.
        print("\n测试集为空，无法输出单张图片预测结果。")
    elif not model_loaded_for_final_eval:
        print("\n没有模型可供评估，无法输出单张图片预测结果。")


    # --- 绘图 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失 (Train Loss)', color='blue')
    
    valid_test_losses = [loss for loss in test_losses if not np.isnan(loss)]
    if valid_test_losses:
         epochs_with_valid_test_loss = [i + 1 for i, loss in enumerate(test_losses) if not np.isnan(loss)]
         plt.plot(epochs_with_valid_test_loss, valid_test_losses, label='测试损失 (Test Loss)', color='red', linestyle='--')

    plt.title('损失随轮次变化 (Loss over Epochs)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失 (Loss)')
    if train_losses or valid_test_losses: # Only show legend if there's data
        plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if train_accuracies:
        plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='训练准确率 (Train Acc)', color='blue')
    
    valid_test_accuracies = [acc for acc in test_accuracies if not np.isnan(acc)]
    if valid_test_accuracies:
        epochs_with_valid_test_acc = [i + 1 for i, acc in enumerate(test_accuracies) if not np.isnan(acc)]
        plt.plot(epochs_with_valid_test_acc, valid_test_accuracies, label='测试准确率 (Test Acc)', color='red', linestyle='--')

    plt.title('准确率随轮次变化 (Accuracy over Epochs)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('准确率 (Accuracy) (%)')
    if train_accuracies or valid_test_accuracies: # Only show legend if there's data
        plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_testing_curves_cnn.png')
    print("训练和测试曲线图已保存为 training_testing_curves_cnn.png")
    plt.show()

if __name__ == '__main__':
    if os.name == 'nt': 
        freeze_support() 
    main()
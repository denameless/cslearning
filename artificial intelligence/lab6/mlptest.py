import numpy as np
import matplotlib.pyplot as plt
import time

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    X = data[:, :4]
    y = data[:, 4].reshape(-1, 1)
    return X, y

X, y = load_data('MLP_data.csv')

# 标准化数据
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
y_mean, y_std = y.mean(), y.std()
X_normalized = (X - X_mean) / X_std
y_normalized = (y - y_mean) / y_std

# 划分数据集
split_index = int(0.8 * len(X_normalized))
X_train, y_train = X_normalized[:split_index], y_normalized[:split_index]
X_test, y_test = X_normalized[split_index:], y_normalized[split_index:]

class MLP:
    def __init__(self, input_size=4, hidden_size=10, output_size=1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y_true, y_pred, lr=0.01):
        n_samples = X.shape[0]
        
        dz2 = (y_pred - y_true) / n_samples
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self._sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def save_best_parameters(self):
        """保存当前参数"""
        self.best_W1 = self.W1.copy()
        self.best_b1 = self.b1.copy()
        self.best_W2 = self.W2.copy()
        self.best_b2 = self.b2.copy()
    
    def load_best_parameters(self):
        """恢复最佳参数"""
        self.W1 = self.best_W1.copy()
        self.b1 = self.best_b1.copy()
        self.W2 = self.best_W2.copy()
        self.b2 = self.best_b2.copy()
    
    def train(self, X_train, y_train, X_test, y_test, epochs=1000, lr=0.01, batch_size=100, patience=10, min_delta=1e-4):
        start_time = time.time()
        train_losses = []
        test_losses = []
        n_samples = X_train.shape[0]   # 训练样本数量

        best_test_loss = float('inf')
        wait = 0

        print(f"Initial Training Loss: {self.compute_loss(y_train, self.forward(X_train)):.4f}")
        print(f"Initial Test Loss: {self.compute_loss(y_test, self.forward(X_test)):.4f}")
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size): # 随机小批量训练
                batch_indices = indices[i:i + batch_size]
                X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]
                
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred, lr)
            
            train_pred = self.forward(X_train)
            test_pred = self.forward(X_test)
            train_loss = self.compute_loss(y_train, train_pred)
            test_loss = self.compute_loss(y_test, test_pred)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

            # Early Stopping监控
            if best_test_loss - test_loss > min_delta:
                best_test_loss = test_loss
                wait = 0
                self.save_best_parameters()  # 保存最佳参数
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}!")
                    break
        
        # 恢复最佳参数
        self.load_best_parameters()
        print("Model parameters restored to the best observed on validation set.")
        
        # 绘制损失曲线
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Testing Loss")
        plt.legend()
        plt.show()
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds.")
    def predict(self, X):
        return self.forward(X)

# 训练模型
mlp = MLP(input_size=4, hidden_size=10)
mlp.train(X_train, y_train, X_test, y_test, epochs=100000, lr=0.005, patience=1000, min_delta=1e-4)

# 预测示例（需标准化输入）
test_sample = np.array([[-122.23, 37.88, 41, 88000]])
test_sample_normalized = (test_sample - X_mean) / X_std
predicted_price_normalized = mlp.predict(test_sample_normalized)

# 反标准化得到实际房价
predicted_price = predicted_price_normalized * y_std + y_mean
print(f"Predicted Price: ${predicted_price[0][0]:.2f}")

# 使用测试集数据
X_vis = X_test * X_std + X_mean  # 反标准化
y_vis = y_test * y_std + y_mean  # 真实房价
y_pred_vis = mlp.predict(X_test) * y_std + y_mean  # 预测房价

# 特征索引
LONGITUDE_IDX = 0
LATITUDE_IDX = 1
HOUSING_AGE_IDX = 2
POPULATION_IDX = 3


longitude = X_vis[:, LONGITUDE_IDX]
latitude = X_vis[:, LATITUDE_IDX]
housing_age = X_vis[:, HOUSING_AGE_IDX]
population = X_vis[:, POPULATION_IDX]

# 点大小根据人口映射
pop_min, pop_max = population.min(), population.max()
sizes = 10 + 40 * (population - pop_min) / (pop_max - pop_min)

# 1. 绘制测试集数据的三维散点图，颜色为真实房价，点大小为人口
fig1 = plt.figure(figsize=(14, 10))
ax1 = fig1.add_subplot(111, projection='3d')
p1 = ax1.scatter(longitude, latitude, housing_age, c=y_vis.flatten(), s=sizes, cmap='viridis', alpha=0.7)
fig1.colorbar(p1, ax=ax1, label='Actual Price')

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_zlabel('Housing Age')
ax1.set_title('Test Set: 3D Scatter Colored by Actual Price')

# 2. 绘制预测房价的三维散点图，颜色为预测房价，点大小为人口
fig2 = plt.figure(figsize=(14, 10))
ax2 = fig2.add_subplot(111, projection='3d')
p2 = ax2.scatter(longitude, latitude, housing_age, c=y_pred_vis.flatten(), s=sizes, cmap='viridis', alpha=0.7)
fig2.colorbar(p2, ax=ax2, label='Predicted Price')

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_zlabel('Housing Age')
ax2.set_title('Test Set: 3D Scatter Colored by Predicted Price')


plt.show()




import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from packing_feature import PackingFeatureExtractor
import os
import glob
from PIL import Image

def preprocess_features(fu_folder):
    """预处理数据集，一次性提取所有特征
    
    Args:
        fu_folder: fu文件夹路径
        
    Returns:
        (state_features_list, action_features_list, utilization_rates): 特征列表和利用率列表
    """
    print('开始预处理数据集，提取所有特征...')
    
    # 初始化特征提取器（只需要一次）
    feature_extractor = PackingFeatureExtractor()
    
    state_features_list = []
    action_features_list = []
    utilization_rates = []
    
    # 遍历所有问题文件夹
    for problem_folder in sorted(glob.glob(os.path.join(fu_folder, 'p*'))):
        if not os.path.isdir(problem_folder):
            continue
            
        print(f'处理文件夹: {problem_folder}')
        
        # 获取所有txt文件
        txt_files = glob.glob(os.path.join(problem_folder, '*.txt'))
        
        for txt_file in txt_files:
            # 获取文件ID
            file_id = os.path.splitext(os.path.basename(txt_file))[0]
            
            # 构建对应的图像文件路径
            state_image = os.path.join(problem_folder, f'{file_id}_s.png')
            action_image = os.path.join(problem_folder, f'{file_id}_a.png')
            
            # 检查文件是否存在
            if os.path.exists(state_image) and os.path.exists(action_image):
                try:
                    # 读取利用率
                    with open(txt_file, 'r') as f:
                        utilization = float(f.read().strip())
                    
                    # 提取特征（一次性完成）
                    state_features = feature_extractor.extract_features_from_image(state_image)
                    state_features = state_features[-1]  # 使用最后一层特征
                    if state_features.ndim > 1:
                        state_features = state_features.flatten()
                    
                    action_features = feature_extractor.extract_features_from_image(action_image)
                    action_features = action_features[-1]  # 使用最后一层特征
                    if action_features.ndim > 1:
                        action_features = action_features.flatten()
                    
                    state_features_list.append(state_features)
                    action_features_list.append(action_features)
                    utilization_rates.append(utilization)
                    
                except (ValueError, IOError) as e:
                    print(f'读取文件 {txt_file} 时出错: {e}')
                    continue
            else:
                print(f'缺少图像文件: {state_image} 或 {action_image}')
    
    print(f'特征提取完成！共处理 {len(state_features_list)} 个样本')
    return state_features_list, action_features_list, utilization_rates

class DualImageDataset(Dataset):
    def __init__(self, state_features, action_features, utilization_rates):
        """双图像输入数据集 - 使用预提取的特征
        
        Args:
            state_features: 预提取的状态特征列表
            action_features: 预提取的动作特征列表
            utilization_rates: 利用率列表
        """
        self.state_features = torch.FloatTensor(state_features)
        self.action_features = torch.FloatTensor(action_features)
        self.utilization_rates = torch.FloatTensor(utilization_rates)
    
    def __len__(self):
        return len(self.state_features)
    
    def __getitem__(self, idx):
        return (
            self.state_features[idx],
            self.action_features[idx],
            self.utilization_rates[idx]
        )

class DualImageMLP(nn.Module):
    def __init__(self, state_dim=512, action_dim=512, hidden_sizes=[512, 256, 128]):
        """双图像输入的MLP模型
        
        Args:
            state_dim: 状态特征维度
            action_dim: 动作特征维度
            hidden_sizes: 隐藏层大小列表
        """
        super(DualImageMLP, self).__init__()
        
        # 状态特征编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
        
        # 动作特征编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
        
        # 融合网络
        input_dim = 256 + 256  # 状态编码 + 动作编码
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # 输出层：预测利用率
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # 利用率在[0,1]范围内
        
        self.fusion_net = nn.Sequential(*layers)
    
    def forward(self, state_features, action_features):
        """前向传播
        
        Args:
            state_features: 状态特征 [batch_size, state_dim]
            action_features: 动作特征 [batch_size, action_dim]
            
        Returns:
            利用率预测值 [batch_size, 1]
        """
        # 编码状态和动作特征
        state_encoded = self.state_encoder(state_features)
        action_encoded = self.action_encoder(action_features)
        
        # 拼接特征
        combined_features = torch.cat([state_encoded, action_encoded], dim=1)
        
        # 预测利用率
        utilization = self.fusion_net(combined_features)
        
        return utilization

class DualImageTrainer:
    def __init__(self, state_dim=512, action_dim=512, hidden_sizes=[512, 256, 128], learning_rate=0.001):
        """双图像训练器"""
        self.model = DualImageMLP(state_dim, action_dim, hidden_sizes)
        self.criterion = nn.MSELoss()  # 回归任务
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def train(self, train_loader, val_loader=None, epochs=100):
        """训练模型"""
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for state_features, action_features, utilization_rates in train_loader:
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(state_features, action_features)
                loss = self.criterion(outputs.squeeze(), utilization_rates)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}')
            
            # 验证
            if val_loader:
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for state_features, action_features, utilization_rates in val_loader:
                        outputs = self.model(state_features, action_features)
                        loss = self.criterion(outputs.squeeze(), utilization_rates)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                print(f'Validation Loss: {avg_val_loss:.6f}')
            
            self.scheduler.step()
    
    def predict_utilization(self, state_features, action_features):
        """预测给定状态-动作特征的利用率"""
        self.model.eval()
        
        # 转换为tensor
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
        action_tensor = torch.FloatTensor(action_features).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            utilization = self.model(state_tensor, action_tensor)
            return utilization.item()
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f'模型已保存到: {path}')
    
    def load_model(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path))
        print(f'模型已从 {path} 加载')

def split_dual_dataset(state_features, action_features, utilization_rates, train_ratio=0.8):
    """划分数据集"""
    total_samples = len(state_features)
    train_size = int(total_samples * train_ratio)
    
    # 随机打乱索引
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 划分数据
    train_state = [state_features[i] for i in train_indices]
    train_action = [action_features[i] for i in train_indices]
    train_util = [utilization_rates[i] for i in train_indices]
    
    val_state = [state_features[i] for i in val_indices]
    val_action = [action_features[i] for i in val_indices]
    val_util = [utilization_rates[i] for i in val_indices]
    
    return train_state, train_action, train_util, val_state, val_action, val_util

if __name__ == '__main__':
    # 1. 预处理数据集，一次性提取所有特征
    fu_folder = '../../dataset/fu'
    print('预处理数据集，提取所有特征...')
    state_features, action_features, utilization_rates = preprocess_features(fu_folder)
    
    # 2. 划分数据集
    print('划分数据集...')
    train_state, train_action, train_util, val_state, val_action, val_util = split_dual_dataset(
        state_features, action_features, utilization_rates, train_ratio=0.8
    )
    
    # 3. 创建数据集和数据加载器
    print('创建数据加载器...')
    train_dataset = DualImageDataset(train_state, train_action, train_util)
    val_dataset = DualImageDataset(val_state, val_action, val_util)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 4. 初始化训练器
    print('初始化模型...')
    print(f'训练样本数: {len(train_dataset)}')
    print(f'验证样本数: {len(val_dataset)}')
    
    trainer = DualImageTrainer(state_dim=512, action_dim=512)
    
    # 5. 开始训练
    print('开始训练...')
    trainer.train(train_loader, val_loader, epochs=50)
    
    # 6. 测试预测
    print('测试预测...')
    if len(state_features) > 0:
        test_utilization = trainer.predict_utilization(state_features[0], action_features[0])
        actual_utilization = utilization_rates[0]
        print(f'预测利用率: {test_utilization:.6f}')
        print(f'实际利用率: {actual_utilization:.6f}')
        print(f'误差: {abs(test_utilization - actual_utilization):.6f}')
    
    # 7. 保存模型
    trainer.save_model('../models/dual_image_utilization_model.pth')
    print('双图像利用率预测模型训练完成并保存！')
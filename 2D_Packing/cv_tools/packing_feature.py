import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms

class PackingFeatureExtractor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        # 初始化预训练的ResNet18模型
        self.model = models.resnet18(pretrained=True)
        # 获取ResNet的各个层
        self.layers = [
            nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu),
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool
        ]
        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features_from_image(self, image_path):
        """从不同卷积层提取特征
        
        Args:
            image_path: 排料结果图像的路径
            
        Returns:
            features_list: 各层提取的特征列表
        """
        # 读取图像并转换为灰度图
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        # 将图像调整到指定大小
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # 转换为三通道图像并提取特征
        image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_tensor = self.transform(image_3ch).unsqueeze(0)
        
        features_list = []
        x = image_tensor
        
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x)
                features_list.append(x.squeeze().numpy())
        
        return features_list
        
    def visualize_features(self, image_path, save_path=None):
        """可视化不同卷积层的特征提取结果
        
        Args:
            image_path: 排料结果图像的路径
            save_path: 保存可视化结果的路径（可选）
        """
        plt.figure(figsize=(15, 8))
        
        # 读取并处理图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # 提取各层特征
        features_list = self.extract_features_from_image(image_path)
        
        # 显示原始图像
        plt.subplot(2, 4, 1)
        plt.title('原始图像')
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
        # 显示各层CNN特征激活图
        layer_names = ['第一层卷积', '残差块1', '残差块2', '残差块3', '残差块4', '平均池化']
        for i, (features, name) in enumerate(zip(features_list, layer_names)):
            plt.subplot(2, 4, i + 2)
            plt.title(name)
            
            # 对特征图进行处理以便可视化
            if i < len(features_list) - 1:  # 非最后一层
                # 选择第一个通道的特征图
                feature_map = features[0] if features.ndim == 3 else features.mean(axis=0)
            # else:  # 最后一层（平均池化层）
            #     # 动态计算特征维度
            #     feature_size = int(np.sqrt(features.size))
            #     feature_map = features.reshape((feature_size, feature_size))
                
            plt.imshow(feature_map, cmap='viridis')
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    image_path = 'pictures/han.png'
    save_path = 'pictures/han_features.png'
    extractor = PackingFeatureExtractor()
    extractor.visualize_features(image_path, save_path)
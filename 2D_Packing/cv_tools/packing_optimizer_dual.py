import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import os
import json
from dual_image_trainer import DualImageMLP, preprocess_features
from packing_feature import PackingFeatureExtractor

import tools.packing as packing
import tools.bottom_left_fill as BLF
from tools.data_process import process_data_xml, process_data_xml_deleteRedundancy
from shapely.geometry import Polygon
from shapely import affinity

class DualImagePackingOptimizer:
    def __init__(self, instance, width, model_path, polys_len, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        基于双图像输入的排料优化器
        
        Args:
            model_path: 训练好的利用率预测模型路径
            device: 计算设备
        """
        self.instance = instance
        self.width = width
        self.device = device
        
        # 使用现有的特征提取器
        self.feature_extractor = PackingFeatureExtractor()
        
        # 加载利用率预测网络
        self.utilization_network = DualImageMLP(state_dim=512, action_dim=512)
        self.utilization_network.load_state_dict(torch.load(model_path, map_location=device))
        self.utilization_network.eval()
        self.utilization_network.to(device)
        
        # 动作空间定义
        self.polygon_indices = range(polys_len)    # 多边形索引
        self.rotation_angles = [0, 30, 45, 60, 90]  # 旋转角度

        path = '../../Data-xml/' + self.instance + '.xml'
        polys_DR = process_data_xml_deleteRedundancy (path)
        self.polys = process_data_xml (path)
        self.nfp_ass = packing.NFPAssistant (polys_DR, store_nfp=True,
                                        store_path='../../record/' + self.instance + '_nfp.csv',
                                        get_all_nfp=True)

    def extract_state_features(self, image_path):
        """
        从图像中提取状态特征
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            state_features: 状态特征向量 (512维)
        """
        # 使用现有的特征提取器
        features_list = self.feature_extractor.extract_features_from_image(image_path)
        
        # 获取最后一层的特征（平均池化层的输出）
        final_features = features_list[-1]  # 最后一层特征
        
        # 确保特征是512维的向量
        if final_features.ndim > 1:
            final_features = final_features.flatten()
        
        # 转换为torch tensor
        state_features = torch.FloatTensor(final_features).to(self.device)
        
        return state_features
    
    def generate_action_image(self, polygon_idx, rotation_angle, image_size=(224, 224)):
        """
        生成动作图像（简化版本，实际应用中需要根据具体需求实现）
        
        Args:
            polygon_idx: 多边形索引
            rotation_angle: 旋转角度
            image_size: 图像尺寸
            
        Returns:
            action_image_path: 生成的动作图像路径
        """
        # 创建临时目录
        temp_dir = 'temp_actions'
        os.makedirs(temp_dir, exist_ok=True)

        tmp_polys = []
        coords = affinity.rotate (Polygon (self.polys[polygon_idx]), rotation_angle).exterior.coords[:-1]
        rotated_poly = [[coord[0], coord[1]] for coord in coords]
        tmp_polys.append (list (rotated_poly))
        bfl = BLF.BottomLeftFill (self.width, tmp_polys, NFPAssistant=self.nfp_ass)
        bfl.textPrint ()
        usage = bfl.patches_area / bfl.board_area
        print ("The Usage Percentage:", usage)
        # 保存临时图像
        action_image_path = temp_dir + '/' + f'action_{polygon_idx}_{rotation_angle}.png'
        bfl.only_savefig (fig_path= action_image_path)

        return action_image_path
    
    def extract_action_features(self, polygon_idx, rotation_angle):
        """
        提取动作特征
        
        Args:
            polygon_idx: 多边形索引
            rotation_angle: 旋转角度
            
        Returns:
            action_features: 动作特征向量 (512维)
        """
        # 生成动作图像
        action_image_path = self.generate_action_image(polygon_idx, rotation_angle)
        
        # 提取特征
        features_list = self.feature_extractor.extract_features_from_image(action_image_path)
        final_features = features_list[-1]
        
        if final_features.ndim > 1:
            final_features = final_features.flatten()
        
        # 清理临时文件
        if os.path.exists(action_image_path):
            os.remove(action_image_path)
        
        return torch.FloatTensor(final_features).to(self.device)
    
    def get_best_action(self, state_features, available_polygons=None):
        """
        根据状态特征获取最佳动作（基于利用率预测）
        
        Args:
            state_features: 状态特征向量
            available_polygons: 可用的多边形索引列表，如果为None则使用所有多边形
            
        Returns:
            best_action: (polygon_index, rotation_angle)
            best_utilization: 对应的预测利用率
        """
        if available_polygons is None:
            available_polygons = self.polygon_indices
            
        best_utilization = float('-inf')
        best_action = None
        
        print(f"评估 {len(available_polygons)} 个多边形 × {len(self.rotation_angles)} 个角度 = {len(available_polygons) * len(self.rotation_angles)} 个动作")
        
        # 遍历所有可能的动作
        for i, polygon_idx in enumerate(available_polygons):
            for j, rotation in enumerate(self.rotation_angles):
                # 提取动作特征
                action_features = self.extract_action_features(polygon_idx, rotation)
                
                # 预测利用率
                with torch.no_grad():
                    utilization = self.utilization_network(
                        state_features.unsqueeze(0), 
                        action_features.unsqueeze(0)
                    )
                    utilization_value = utilization.item()
                
                if utilization_value > best_utilization:
                    best_utilization = utilization_value
                    best_action = [polygon_idx, rotation]
                
                # 显示进度
                if (i * len(self.rotation_angles) + j + 1) % 20 == 0:
                    print(f"已评估 {i * len(self.rotation_angles) + j + 1} 个动作...")
        
        return best_action, best_utilization
    
    def get_action_sequence(self, initial_image_path, max_polygons=30, output_dir=None):
        """
        获取最优动作序列（基于利用率预测）
        
        Args:
            initial_image_path: 初始状态图像路径
            max_polygons: 最大多边形数量
            output_dir: 输出目录，用于保存中间状态图像（可选）
            
        Returns:
            action_sequence: 动作序列列表
            utilization_values: 对应的预测利用率列表
        """
        action_sequence = []
        utilization_values = []
        used_polygons = set()
        
        current_image_path = initial_image_path
        
        print(f"开始优化排料序列，初始图像: {initial_image_path}")
        print(f"使用基于利用率预测的双图像模型")

        # 提取当前状态特征
        state_features = self.extract_state_features (current_image_path)
        for step in range(max_polygons):
            print(f"\n=== 步骤 {step + 1} ===")

            
            # 获取可用的多边形（排除已使用的）
            available_polygons = [p for p in self.polygon_indices if p not in used_polygons]
            
            if not available_polygons:
                print(f"所有多边形已使用完毕，在第 {step} 步结束")
                break
            
            # 获取最佳动作
            best_action, best_utilization = self.get_best_action(state_features, available_polygons)
            
            if best_action is None:
                print(f"无法找到有效动作，在第 {step} 步结束")
                break
            
            polygon_idx, rotation = best_action
            used_polygons.add(polygon_idx)
            
            action_sequence.append(best_action)
            utilization_values.append(best_utilization)
            
            print(f"选择多边形 {polygon_idx}, 旋转 {rotation}°, 预测利用率: {best_utilization:.4f}")
            
            # 注意：这里需要您实现实际的排料环境来生成下一个状态图像
            # 目前只是示例，实际使用时需要调用排料环境
            # current_image_path = self.apply_action_and_get_next_state(current_image_path, best_action)

            ##保存当前动作之后生成的图片
            tmp_polys = []
            for (idx, angle) in action_sequence:
                coords = affinity.rotate (Polygon (self.polys[idx]), angle).exterior.coords[:-1]
                rotated_poly = [[coord[0], coord[1]] for coord in coords]
                tmp_polys.append (list (rotated_poly))
            bfl = BLF.BottomLeftFill (self.width, tmp_polys, NFPAssistant=self.nfp_ass)
            bfl.textPrint ()
            usage = bfl.patches_area / bfl.board_area
            print ("The Usage Percentage:", usage)
            bfl.only_savefig (fig_path='mid_png/' + str (step) + '.png')
            # 通过排料方法来生成下一个状态图像,更新图片，重新提取状态特征

            current_image_path = 'mid_png/' + str (step) + '.png'
            state_features = self.extract_state_features (current_image_path)

        return action_sequence, utilization_values


def main():
    """
    主函数示例
    """
    instance = "fu"
    width = 38
    # 初始化优化器
    model_path = '../models/dual_image_utilization_model.pth'  # 双图像利用率预测模型路径
    poly_len = 12
    optimizer = DualImagePackingOptimizer(instance, width, model_path, poly_len)

    # 多边形3, 旋转0°
    path = '../../Data-xml/' + instance + '.xml'
    polys_DR = process_data_xml_deleteRedundancy (path)
    polys = process_data_xml (path)
    nfp_ass = packing.NFPAssistant (polys_DR, store_nfp=True,
                                         store_path='../../record/' + instance + '_nfp.csv',
                                         get_all_nfp=True)

    tmp_polys = []
    coords = affinity.rotate (Polygon (polys[3]), 0).exterior.coords[:-1]
    rotated_poly = [[coord[0], coord[1]] for coord in coords]
    tmp_polys.append (list (rotated_poly))
    bfl = BLF.BottomLeftFill (width, tmp_polys, NFPAssistant=nfp_ass)
    bfl.textPrint ()
    # 保存临时图像
    bfl.only_savefig (fig_path='start_state.png')
    # 输入图像路径
    initial_image = 'start_state.png'
    # initial_image = '../../dataset/fu/p12/1_s.png'  # 初始状态图像,

    # 获取最优动作序列
    action_sequence, utilization_values = optimizer.get_action_sequence(
        initial_image_path=initial_image,
        max_polygons=poly_len  # 限制最大步数用于测试
    )
    
    print(f"\n=== 最优动作序列 ({len(action_sequence)} 步) ===")
    for i, (action, util_val) in enumerate(zip(action_sequence, utilization_values)):
        polygon_idx, rotation = action
        print(f"步骤 {i+1}: 多边形 {polygon_idx}, 旋转 {rotation}°, 预测利用率: {util_val:.4f}")

    # 清理临时文件夹
    import shutil
    if os.path.exists('temp_actions'):
        shutil.rmtree('temp_actions')
        print("\n已清理临时文件")

if __name__ == '__main__':
    main()


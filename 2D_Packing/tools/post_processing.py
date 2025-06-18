from shapely.geometry import Polygon
import numpy as np

class PostProcessing:
    @staticmethod
    def check_overlap(polygons):
        """
        检测多边形之间是否存在重叠
        Args:
            polygons: 多边形列表，每个元素是一个二维数组表示的多边形顶点坐标
        Returns:
            overlaps: 重叠的多边形对的列表，每个元素是一个元组 (i, j, overlap_area)
        """
        overlaps = []
        for i in range(len(polygons)):
            poly1 = Polygon(polygons[i])
            for j in range(i + 1, len(polygons)):
                poly2 = Polygon(polygons[j])
                if poly1.intersects(poly2):
                    intersection = poly1.intersection(poly2)
                    if intersection.area > 1e-10:  # 考虑浮点数精度，设置一个小的阈值
                        overlaps.append((i, j, intersection.area))
        return overlaps

    @staticmethod
    def remove_overlap(polygons):
        """
        处理多边形之间的重叠，将重叠部分从面积较小的多边形中剔除
        Args:
            polygons: 多边形列表，每个元素是一个二维数组表示的多边形顶点坐标
        Returns:
            processed_polygons: 处理后的多边形列表，三维数组格式 [[[x1,y1], [x2,y2], ...], ...]
        """
        processed_polygons = []
        for i, poly in enumerate(polygons):
            current_poly = Polygon(poly)
            # 检查与其他多边形的重叠
            for j, other_poly in enumerate(polygons):
                if i != j:
                    other_shape = Polygon(other_poly)
                    if current_poly.intersects(other_shape):
                        if current_poly.area <= other_shape.area:
                            current_poly = current_poly.difference(other_shape)
            
            # 将处理后的多边形转换回坐标列表格式
            if current_poly.geom_type == 'Polygon':
                coords = list(current_poly.exterior.coords)[:-1]  # 去掉最后一个重复的点
                # 转换为 [x,y] 格式的列表
                coords = [[p[0], p[1]] for p in coords]
                processed_polygons.append(coords)
            elif current_poly.geom_type == 'MultiPolygon':
                largest = max(current_poly.geoms, key=lambda p: p.area)
                coords = list(largest.exterior.coords)[:-1]
                # 转换为 [x,y] 格式的列表
                coords = [[p[0], p[1]] for p in coords]
                processed_polygons.append(coords)

        return processed_polygons

    @staticmethod
    def visualize_comparison(original_polygons, processed_polygons):
        """
        可视化对比处理前后的多边形排布
        Args:
            original_polygons: 原始多边形列表
            processed_polygons: 处理后的多边形列表
        """
        # 创建两个子图
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # 绘制原始排布
        ax1.set_title('Original Layout')
        for poly in original_polygons:
            polygon = Polygon(poly)
            x, y = polygon.exterior.xy
            ax1.fill(x, y, alpha=0.5)

        # 绘制处理后的排布
        ax2.set_title('Processed Layout')
        for poly in processed_polygons:
            polygon = Polygon(poly)
            x, y = polygon.exterior.xy
            ax2.fill(x, y, alpha=0.5)

        plt.show()

# 测试代码
if __name__ == '__main__':
    # 创建一个简单的测试用例，包含重叠的多边形
    test_polygons = [
        [[0, 0], [2, 0], [2, 2], [0, 2]],  # 正方形1
        [[1, 1], [3, 1], [3, 3], [1, 3]],  # 正方形2，与正方形1重叠
        [[4, 0], [5, 2], [4, 4], [3, 2]]    # 凹多边形
    ]
    # 检测重叠
    overlaps = PostProcessing.check_overlap(test_polygons)
    print("检测到的重叠:", overlaps)

    # 处理重叠
    processed_polygons = PostProcessing.remove_overlap(test_polygons)
    print("处理后的多边形:", processed_polygons)
    
    # 可视化对比
    PostProcessing.visualize_comparison(test_polygons, processed_polygons)
import pyclipper
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

def scale_polygon(polygon, scale_factor):
    """改进的缩放函数，使用round避免直接截断造成的误差"""
    return [[int(round(x * scale_factor)), int(round(y * scale_factor))] for x, y in polygon]

def unscale_polygon(polygon, scale_factor):
    """ 计算后将整数坐标缩小回原来的比例 """
    return [[x / scale_factor, y / scale_factor] for x, y in polygon]

def clean_polygon(polygon, scale_factor):
    """ 使用布尔运算清理多边形，处理自相交等问题 """
    # 缩放并转换为Clipper格式
    scaled_poly = scale_polygon(polygon, scale_factor)
    pc = pyclipper.Pyclipper()
    pc.AddPath(scaled_poly, pyclipper.PT_SUBJECT, True)
    
    # 使用Union操作清理多边形
    cleaned = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
    
    # 如果结果为空，返回原始多边形
    if not cleaned:
        return [polygon]
    
    # 还原比例并返回
    return [unscale_polygon(p, scale_factor) for p in cleaned]

def get_optimal_scale_factor(poly1, poly2):
    """计算最优的缩放因子，同时考虑精度和溢出问题"""
    max_decimal_places = 0
    max_coord = 0
    
    # 检查所有坐标点的小数位数和最大坐标值
    for polygon in [poly1, poly2]:
        for x, y in polygon:
            # 检查最大坐标值
            max_coord = max(max_coord, abs(x), abs(y))
            # 检查小数位数
            decimal_x = len(str(float(x)).split('.')[-1]) if '.' in str(float(x)) else 0
            decimal_y = len(str(float(y)).split('.')[-1]) if '.' in str(float(y)) else 0
            max_decimal_places = max(max_decimal_places, decimal_x, decimal_y)
    
    # 计算安全的缩放因子
    scale_factor = 10 ** max_decimal_places
    
    # 检查是否会溢出，如果会溢出则调整缩放因子
    while max_coord * scale_factor > 1e9:  # 设置一个安全阈值
        scale_factor /= 10
        max_decimal_places -= 1
    
    return max(scale_factor, 1e3)  # 确保最小缩放因子不小于1000，保持基本精度

def nfp_polygon_min(poly1, poly2):
    """
    计算 Minkowski Sum 适用于非凸多边形
    poly1: 固定多边形 (list of lists)
    poly2: 运动多边形 (list of lists, 需取负)
    返回: 计算后的 No-Fit Polygon (list of list of lists)
    """
    # 计算最优的缩放因子
    scale_factor = get_optimal_scale_factor(poly1, poly2)

    # 预处理：清理输入多边形
    poly1_cleaned = clean_polygon(poly1, scale_factor)[0]  # 取第一个结果
    poly2_cleaned = clean_polygon(poly2, scale_factor)[0]

    # 缩放多边形（转换为整数坐标）
    poly1_scaled = scale_polygon(poly1_cleaned, scale_factor)
    poly2_scaled = scale_polygon([[-x, -y] for x, y in poly2_cleaned], scale_factor)  # 取负形

    # 使用 Clipper 计算 Minkowski Sum
    result = pyclipper.MinkowskiSum(poly1_scaled, poly2_scaled, pyclipper.PFT_NONZERO)

    # 对结果进行布尔Union操作，确保结果的正确性
    pc = pyclipper.Pyclipper()
    for path in result:
        pc.AddPath(path, pyclipper.PT_SUBJECT, True)
    final_result = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

    # 还原到原始比例
    result_unscaled = [unscale_polygon(p, scale_factor) for p in final_result]
    
    # 过滤内部NFP
    filtered_result = []
    for polygon in result_unscaled:
        # 使用Shapely创建多边形对象
        poly = Polygon(polygon)
        # 检查多边形的方向（顺时针为外部NFP）
        if poly.exterior.is_ccw:  # 外部NFP应该是逆时针方向的 counter-clockwise winding
            filtered_result.append(polygon)

    print("使用minkowski方法得到NFP")
    return filtered_result # 返回所有外部NFP，操作的时候默认返回第一个外部NFP

if __name__ == "__main__":
    # 示例：定义非凸多边形
    # poly_A = [[0, 0], [5, 0], [3, 2], [4, 4], [1, 3]]  # 非凸多边形
    # poly_B = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]  # L形非凸多边形
    
    # # Теst 1
    # poly_A = [[6, 2], [8, 4], [10, 2], [8, 0]]
    # poly_B = [[0, 0], [2, 2], [4, 0]]
    # Test 2    
    # poly_A = [[5, 7], [8, 7], [8, 5], [6, 5], [6, 3], [8, 3], [8, 1], [10, 1],
    #           [10, 3], [12, 3], [12, 5], [10, 5], [10, 7], [13, 7], [13, 0], [5, 0]]
    # poly_B = [[0, 2], [2, 2], [2, 0], [0, 0]]
    # # Test 3
    # poly_A = [[0, 0], [0, 6], [5, 5.5], [7, 3.5], [3, 5], [1, 2.5], [4, 1], [6, 2], [6, 0]]
    # poly_B = [[-6, -3], [-5, -1], [-2, -1], [-1, -3.5], [-2, -2], [-4, -1.5]]
    # # Test 4
    # poly_A = [[6, 2], [6, 4], [8, 4], [8, 2]]
    # poly_B = [[0, 0], [0, 2], [2, 2], [2, 0]]
    # # Test 5
    poly_A = [[0.0, 2.0], [7.0, 0.0], [8.0, 2.0], [8.0, 3.0], [9.0, 4.0], [8.0, 5.0], [7.0, 7.0], [0.0, 6.0]]
    poly_B = [[0.0, -7.0], [1.0, -8.0], [2.0, -10.0], [9.0, -9.0], [9.0, -5.0], [2.0, -3.0], [1.0, -5.0], [1.0, -6.0]]

    # 计算 No-Fit Polygon
    nfp_result = nfp_polygon_min(poly_A, poly_B)
    print(nfp_result)
    # 可视化
    fig, ax = plt.subplots()
    ax.plot(*zip(*poly_A, poly_A[0]), 'bo-', label="Polygon A (Fixed)")
    ax.plot(*zip(*poly_B, poly_B[0]), 'go-', label="Polygon B (Moving)")

    # 绘制 No-Fit Polygon 结果
    for nfp in nfp_result:
        ax.plot(*zip(*nfp, nfp[0]), 'r-', label="No-Fit Polygon")###加入nfp[0]，闭合多边形

    ax.legend()
    ax.set_aspect('equal')# 设置坐标轴比例为等比例
    plt.show()

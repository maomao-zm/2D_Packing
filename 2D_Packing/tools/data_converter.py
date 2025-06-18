def array_to_dict(points_array):
    """
    输入多边形格式修改
    将二维数组形式的多边形坐标点转换为字典形式
    
    Args:
        points_array: 二维数组，形如 [[x1,y1], [x2,y2], ...]
        
    Returns:
        dict: 包含points列表的字典，形如 {'points':[{'x':x1, 'y':y1}, {'x':x2, 'y':y2}, ...]}
    """
    points_list = []
    for point in points_array:
        if len(point) != 2:
            raise ValueError("每个点必须包含且仅包含x和y两个坐标值")
        points_list.append({'x': point[0], 'y': point[1]})
    
    return {'points': points_list}

def dict_to_array(polygons):
    """
    NFP格式修改
    将字典形式的多边形坐标点转换为三维数组形式
    
    Args:
        polygons: 字典列表数组，形如 [[{'x':x1, 'y':y1}, {'x':x2, 'y':y2}, ...], ...]
        
    Returns:
        list: 三维数组，形如 [[[x1,y1], [x2,y2], ...], ...]
    """
    result = []
    for polygon in polygons:
        points = []
        for point in polygon:
            points.append([point['x'], point['y']])
        result.append(points)
    return result

# 测试代码
if __name__ == '__main__':
    # 原有的array_to_dict测试代码
    test_array = [[6,2], [8,4], [10,2], [8,0]]
    result = array_to_dict(test_array)
    print("输入数组：", test_array)
    print("转换结果：", result)
    expected = {'points':[{'x': 6, 'y':2}, {'x': 8, 'y':4}, {'x': 10, 'y':2}, {'x': 8, 'y':0}]}
    print("验证结果：", result == expected)
    
    # dict_to_array测试代码
    test_dict = [[{'x': 8, 'y': 0}, {'x': 10, 'y': 2}, {'x': 12, 'y': 4}, 
                 {'x': 10, 'y': 6}, {'x': 6, 'y': 6}, {'x': 4, 'y': 4}, {'x': 6, 'y': 2}]]
    array_result = dict_to_array(test_dict)
    print("\n字典转数组测试：")
    print("输入字典：", test_dict)
    print("转换结果：", array_result)
    expected_array = [[[8,0], [10,2], [12,4], [10,6], [6,6], [4,4], [6,2]]]
    print("验证结果：", array_result == expected_array)
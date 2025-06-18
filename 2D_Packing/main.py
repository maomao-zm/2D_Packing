'''
将所有packing方法统一接口, class封装
0，reward 原采用 高度的倒数且完整序列packing之后reward
改进点：
1，放入每一个patch之后reward
2, 加入旋转，将action作为为一个（idx，angle）tuple
3，改进点：使用物理仿真 physical simulate by pymunk
通过完成每次全局放置之后，通过物理仿真加入，或许有更好的效果
4，使用DQN代替传统方法
5, 综合以上方法, 舍弃即时反馈，性价比很低
'''
import numpy as np
import tools.packing as packing
import tools.bottom_left_fill as BLF
from tools.data_process import process_data_xml, process_data_xml_deleteRedundancy
from tools.assistant import OutputFunc as OPF
import time

import solve_Q_learning as S0
import solve_rotation as S1
import solve_Q_improveReward as S2
import solve_phyiscalSimulator as S3
import solve_DQN as S4
import solve_ours as S5
from openpyxl import load_workbook

from shapely import affinity
from shapely.geometry import Polygon

def main(name, width, idx, max_episode):
    '''其中参数更改主要为 name, width and idx 即不同的RL方法'''
    # EPSILON = 0.9  # greedy police
    # ALPHA = 0.5  # learning rate
    # GAMMA = 1.  # discount factor
    # MAX_EPISODES = 300  # maximum episodes
    # C = 100  ## the constant

    # name = 'dighe2'
    # width = 105
    # idx = 2  ###表示RL方法 index

    start = time.time()

    methods = [S0, S1, S2, S3, S4, S5]
    times = 10  ##一共实验十次，取最好和平均

    usage_list = []
    polys_idx_list = []

    path = '../Data-xml/' + name + '.xml'
    polys_DR = process_data_xml_deleteRedundancy(path)
    # packing.NFPAssistant(polys_DR, store_nfp=True, store_path='../record/' + name + '_nfp.csv', get_all_nfp_xml=True,
    #                      xml_path=path)  ###第一次将NFP写入当地文件. 默认输入去重之后的polys_DR
    packing.NFPAssistant(polys_DR, store_nfp=True, store_path='../record/' + name + '_nfp.csv', get_all_nfp=True)
    compare_usage = 0. ##全局times usage ,最后得到最好的usage
    print("使用" + str(methods[idx]) + "方法")
    for i in range(times):
        OPF.outputInfo('times: ' + str(i + 1) + ' ', '')
        sql = methods[idx].Solve(name, width, MAX_EPISODES=max_episode, compare_usage=compare_usage)
        q_table = sql.rl()
        if q_table is not None:  ### DQN return None
            print('\r\nQ-table:\n')
            print(q_table)
        usage_list.append(sql.best_usage)
        polys_idx_list.append(sql.best_polys_idx)
        compare_usage = sql.compare_usage

    end = time.time()

    ## plot the best result
    print("plot the best result")
    polys = process_data_xml(path)
    # 处理最佳结果的多边形列表

    best_idx = np.argmax(usage_list)
    tmp_polys = []
    # 检查第一个元素的类型来判断数据结构
    if not isinstance(polys_idx_list[best_idx][0], tuple):
        # 如果不是元组，说明是简单的索引数组
        for idx in polys_idx_list[best_idx]:
            tmp_polys.append(polys[idx])
    else:
        # 如果是元组，说明包含旋转信息
        for polys_idx in polys_idx_list[best_idx]:
            idx, angle = polys_idx  # 从元组中获取索引和角度
            coords = affinity.rotate(Polygon(polys[idx]), angle).exterior.coords[:-1]
            rotated_poly = [[coord[0], coord[1]] for coord in coords]
            tmp_polys.append(list(rotated_poly))

    nfp_ass = packing.NFPAssistant (polys_DR, load_history=True, history_path='../record/' + name + '_nfp.csv')
    blf = BLF.BottomLeftFill (width, tmp_polys, NFPAssistant=nfp_ass)
    blf.textPrint()
    blf.showAll()
    ###精度问题导致有误差，所以直接在算法中生成图片并save，此处仅展示
    print('usage list', usage_list)
    print('Best Usage polys idx: ', polys_idx_list[np.argmax(usage_list)])
    print('Best Usage Result: ', compare_usage, np.max(usage_list), blf.patches_area/blf.board_area)
    print('Average Usage Result: ', np.sum(usage_list) / times)
    print('Total time seconds: ', end - start)

if __name__ == '__main__':
    '''其中参数更改主要为 name, width and idx 即不同的RL方法'''
    '''total number of patches < 100 and >= 50, MAX_EPISODES=50. < 50 and > 30, MAX_EPISODES=100. else <= 30, MAX_EPISODES=300'''
    '''shapes1和shapes0忽视旋转则相同, 采用libnfporb_interface'''
    name = 'jakobs1'
    width = 13
    idx = 5 ###表示RL方法 index
    max_episode = 300
    main(name, width, idx, max_episode)
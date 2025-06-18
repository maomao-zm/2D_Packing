import tools.bottom_left_fill as BLF
import pandas as pd
import numpy as np
import copy
from tools.data_process import process_data_xml, process_data_xml_deleteRedundancy
import tools.packing as packing
from shapely.geometry import Polygon

def greedy_order(polys):
    """
    贪心得到序列索引，面积从大到小排序
    :param polys: 输入多边形数据
    :return: 按面积从大到小排序的多边形索引顺序
    """
    area_order = []

    # 计算每个多边形的面积，并记录下每个多边形的索引
    for i in range(len(polys)):
        area_order.append((Polygon(polys[i]).area, i))  # 记录面积和索引

    # 按照面积从大到小排序，排序后只取出索引部分
    area_order.sort(reverse=True, key=lambda x: x[0])  # 根据面积进行排序
    order = [x[1] for x in area_order]  # 取出排序后的索引

    return order

def get_result(name, width):
    fig_path = 'best_result_picture/' + name + '/greedy.png'
    path = '../Data-xml/' + name + '.xml'
    polys_DR = process_data_xml_deleteRedundancy(path)
    polys = process_data_xml(path)
    nfp_ass = packing.NFPAssistant(polys_DR, load_history=True, history_path='../record/' + name + '_nfp.csv')
    order = greedy_order(polys)
    blf = BLF.BottomLeftFill(width, [polys[i] for i in order], NFPAssistant=nfp_ass)
    blf.textPrint()
    usage = blf.patches_area / blf.board_area
    print("The Usage Percentage:", usage)
    blf.only_savefig(fig_path=fig_path)
    blf.showAll()

if __name__ == "__main__":
    name_list = ["albano", "Blaz", "dighe1", "dighe2", "fu", "Han", "jakobs1", "jakobs2", "marques", "shapes1", "shirts", "swim", "trousers"]
    width_list = [4900, 15, 100, 100, 38, 58, 40, 70, 104, 40, 40, 5752, 79]
    for i in range(len(name_list)):
        get_result(name_list[i], width_list[i])
"""
该文件实现了主要基于序列的排样算法
而怎样优化序列则成为问题所在
"""
from tools.geofunc import GeoFunc
from tools.show import PltFunc
import tools.packing as packing
from tools.nfp import NFP
from shapely.geometry import Polygon,mapping
from shapely import affinity
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import json
import csv
import time
import multiprocessing
import datetime
import random
import copy

from tools.nfp_minkowski import nfp_polygon_min ##minkowski ways
from tools.post_processing import PostProcessing


class BottomLeftFill(object):
    '''
    self.length = self.contain_length if self.contain_length > self.width else self.width
    self.vertical = True 表示先bottom，后left
    self.vertical = False 表示先left，后bottom
    '''
    def __init__(self,width,original_polygons,**kw):
        self.choose_nfp=False
        self.width=width###接口，vertical == True 则表示board宽度，vertical == False 则表示board高度
        self.length=150000 # 在得到IFP中需要的控制变量，无实际意义
        self.contain_height=0 ## vertical == True 则表示最终排料的board高度
        self.contain_width=0 ## vertical == False 则表示最终排料的board宽度
        self.polygons=original_polygons
        self.NFPAssistant=None

        self.libnfpmin_interface = True
        if 'libnfpmin_interface' in kw:
            self.libnfpmin_interface=kw["libnfpmin_interface"]

        if 'NFPAssistant' in kw:
            self.NFPAssistant=kw["NFPAssistant"]
        self.vertical=True###默认垂直情况下
        if 'vertical' in kw:
            self.vertical=kw['vertical']
        if 'xml_path' in kw:
            self.xml_path = kw['xml_path']

        self.placeFirstPoly()
        for i in range(1, len(self.polygons)):
            self.placePoly(i)

        self.getLength()
        # self.showAll()
        self.patches_area = 0
        for i in self.polygons:
            self.patches_area += Polygon(i).area
        if self.vertical == True:
            self.board_area = self.width * self.contain_height
        else:
            self.board_area = self.contain_width * self.width
    def textPrint(self):
        print("Total Num:", len(self.polygons))
        # for i in range(0, len(self.polygons)):
        #     print("############################## Place the ", i + 1, "th shape #################################")

    def placeFirstPoly(self):
        poly=self.polygons[0]
        left_index,bottom_index,right_index,top_index=GeoFunc.checkBound(poly) # 获得边界
        GeoFunc.slidePoly(poly,-poly[left_index][0],-poly[bottom_index][1]) # 平移到左下角

    def placePoly(self,index):

        adjoin=self.polygons[index]
        # 选取垂直
        if self.vertical==True:
            ifr=packing.PackingUtil.getInnerFitRectangle(self.polygons[index],self.width,self.length) ##adjoin发生变动
            # ifr=packing.PackingUtil.getInnerFitRectangle_xml(self.xml_path, self.polygons[index])
        else:
            ifr=packing.PackingUtil.getInnerFitRectangle(self.polygons[index],self.length,self.width)
            # ifr = packing.PackingUtil.getInnerFitRectangle_xml(self.xml_path, self.polygons[index])
        differ_region=Polygon(ifr) #get the Inner fit rectangle -> Polygon

        for main_index in range(0,index):
            main=self.polygons[main_index]
            if self.NFPAssistant==None:
                    '''
                    默认只取最外层polygon，且保持逆时针，且封闭多边形  
                    '''
                    if self.libnfpmin_interface == True:
                        nfp = nfp_polygon_min(main, adjoin)[0]
                    else:
                        nfp = NFP(main,adjoin).nfp
            else: ##从文件中拿到的nfp,此nfp需要进行移动
                nfp=self.NFPAssistant.getDirectNFP(main,adjoin)

            ##nfp移动到main的bottom处 the most important
            start_point_index = GeoFunc.checkBottom(main)  ## static polygon bottom point
            start_point=[main[start_point_index][0],main[start_point_index][1]]
            nfp_start_point_index = GeoFunc.checkBottom(nfp)  ## static polygon bottom point
            nfp_start_point = [nfp[nfp_start_point_index][0], nfp[nfp_start_point_index][1]]
            GeoFunc.slideToPoint(nfp, nfp_start_point, start_point)###规范化, bottom point start point 与NFP算法保持一致

            nfp_poly=Polygon(nfp) ##得到nfp-polygon

            # PltFunc.addPolygon(main)
            # PltFunc.addPolygon(adjoin)
            # PltFunc.addPolygonColor(nfp)
            # PltFunc.showPlt()
            try:
                # Returns the part of geometry A that does not intersect with geometry B.
                # Returns the difference of the geometries.Refer to `shapely.difference` for full documentation.
                differ_region=differ_region.difference(nfp_poly)
            except:
                print('NFP failure, areas of polygons are:')
                self.showAll()
                for poly in main,adjoin:
                    print(Polygon(poly).area)
                self.showPolys([main]+[adjoin]+[nfp])
                if self.NFPAssistant != None:
                    print('NFP loaded from: ',self.NFPAssistant.history_path)

        differ=GeoFunc.polyToArr(differ_region) ## to list
        differ_index=self.getBottomLeft(differ)
        refer_pt_index=GeoFunc.checkTop(adjoin) ## NFP和IFP的reference point为checkTop,这二者与此方法中的reference point一定保持一致
        # GeoFunc.slideToPoint(self.polygons[index],adjoin[refer_pt_index],differ[differ_index]) ##ERROR
        self.polygons[index] = GeoFunc.getSlideToPoint(self.polygons[index],adjoin[refer_pt_index],differ[differ_index])

    def getBottomLeft(self,poly):
        '''
        获得左底部点，优先左侧，有多个左侧选择下方 vertical == False
        或者优先下方，有多个下方选择最左侧 vertical == True
        '''
        bl=[] # bottom left的全部点
        _min=999999
        # 选择最左侧的点
        for i,pt in enumerate(poly):
            pt_object={
                    "index":i,
                    "x":pt[0],
                    "y":pt[1]
            }
            if self.vertical==True:
                target=pt[1]
            else:
                target=pt[0]
            if target<_min:
                _min=target
                bl=[copy.deepcopy(pt_object)]
            elif target==_min:
                bl.append(copy.deepcopy(pt_object))

        if len(bl)==1:
            return bl[0]["index"]
        else:
            if self.vertical==True:
                target="x"                
            else:
                target="y"
            _min=bl[0][target]
            one_pt=bl[0]
            for pt_index in range(1,len(bl)):
                if bl[pt_index][target]<_min:
                    one_pt=bl[pt_index]
                    _min=one_pt[target] ## "y" -> target
            return one_pt["index"]

    def showAll(self, *args, **kwargs):
        # for i in range(0,2):
        for i in range(0,len(self.polygons)):
            PltFunc.addPolygon_fill(self.polygons[i])
            # PltFunc.addPolygon(self.polygons[i])

        if 'fig_path' in kwargs and 'only_save' in args: ##only save fig
            if self.vertical == True:
                PltFunc.showPlt('only_save', width=self.width, height=self.contain_height, fig_path=kwargs['fig_path'])
            else:
                PltFunc.showPlt('only_save', width=self.contain_width, height=self.width, fig_path=kwargs['fig_path'])
            return

        if self.vertical == True:
            if 'fig_path' in kwargs:
                if 'usage' in kwargs:
                    PltFunc.showPlt(width=self.width, height=self.contain_height, fig_path=kwargs['fig_path'], usage=kwargs['usage'])
                else:
                    PltFunc.showPlt(width=self.width, height=self.contain_height, fig_path=kwargs['fig_path'])
            elif 'usage' in kwargs:
                PltFunc.showPlt(width=self.width,height=self.contain_height, usage=kwargs['usage'])
            else:
                PltFunc.showPlt(width=self.width, height=self.contain_height)
        else:
            if 'fig_path' in kwargs:
                if 'usage' in kwargs:
                    PltFunc.showPlt(width=self.contain_width, height=self.width, fig_path=kwargs['fig_path'],
                                    usage=kwargs['usage'])
                else:
                    PltFunc.showPlt(width=self.contain_width, height=self.width, fig_path=kwargs['fig_path'])
            elif 'usage' in kwargs:
                PltFunc.showPlt(width=self.contain_width, height=self.width, usage=kwargs['usage'])
            else:
                PltFunc.showPlt(width=self.contain_width, height=self.width)

    def showPolys(self,polys):
        for i in range(0,len(polys)-1):
            PltFunc.addPolygon(polys[i])
        PltFunc.addPolygonColor(polys[len(polys)-1])
        if self.vertical == True:
            PltFunc.showPlt(width=self.width, height=self.contain_height)
        else:
            PltFunc.showPlt(width=self.contain_width, height=self.width)

    def getLength(self): ## return contain_height or contain_width
        _max=0
        for i in range(0, len(self.polygons)):
            if self.vertical == True:## return height
                extreme_index = GeoFunc.checkTop(self.polygons[i])
                extreme = self.polygons[i][extreme_index][1]
            else: ## return width
                extreme_index = GeoFunc.checkRight(self.polygons[i])
                extreme = self.polygons[i][extreme_index][0]
            if extreme > _max:
                _max = extreme

        if self.vertical == True:
            self.contain_height = _max
        else:
            self.contain_width = _max
        return _max

    def only_savefig(self, fig_path): ## only save figure, no show figure
        self.showAll('only_save', fig_path=fig_path)

if __name__=='__main__':
    ### test code
    polys = [[[1, 1], [2, 1], [1.5, 1.5]],[[0, 0], [1, 0.5], [0, 1]]] ## polygon逆时针
    starttime = datetime.datetime.now()
    bfl=BottomLeftFill(5,polys,vertical=True, libnfpmin_interface=False)

    endtime = datetime.datetime.now()
    print ("total time: ",endtime - starttime)
    print(bfl.contain_height)
    bfl.showAll()
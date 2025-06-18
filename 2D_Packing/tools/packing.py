from tools.nfp import NFP
from tools.show import PltFunc
from shapely.geometry import Polygon,Point,mapping,LineString
from shapely import affinity
from tools.geofunc import GeoFunc
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import csv
import random
import copy
from multiprocessing import Pool
import tools.bottom_left_fill as BLF
import xml.dom.minidom as xmldom
from tools.data_process import process_data_xls, process_data_xml, process_data_xml_deleteRedundancy
import os
from tools.nfp_minkowski import nfp_polygon_min ##minkowski ways

def getNFP(poly1,poly2): # 这个函数必须放在class外面否则多进程报错
    nfp=NFP(poly1,poly2).nfp
    return nfp

class Poly(object):
    '''
    用于后续的Poly对象
    '''
    def __init__(self,num,poly,allowed_rotation):
        self.num=num
        self.poly=poly
        self.cur_poly=poly ##mark, recursive
        self.allowed_rotation=[0,180]

class PackingUtil(object):

    @staticmethod
    def getInnerFitRectangle(poly,x,y):
        '''
        选择一个形状poly加入，通过计算inner fit polygon，也就是形状绕着board内部一周，参考点P会形成的一个长方形，
        P点在该长方形内部则解是feasible solution, 即认为此patch在board中
        此方法中P点取最上方点作为reference point, check_top
        :param poly:
        :param x: board width
        :param y: board Height
        :return: Inner fit rectangle
        '''
        left_index,bottom_index,right_index,top_index=GeoFunc.checkBound(poly) # 获得边界
        new_poly=GeoFunc.getSlide(poly,-poly[left_index][0],-poly[bottom_index][1]) # 平移到左下角

        ##reference point
        refer_pt=[new_poly[top_index][0],new_poly[top_index][1]]
        ifr_width=x-new_poly[right_index][0]
        ifr_height=y-new_poly[top_index][1]

        ##rectangle
        IFR=[refer_pt,[refer_pt[0]+ifr_width,refer_pt[1]],[refer_pt[0]+ifr_width,refer_pt[1]+ifr_height],[refer_pt[0],refer_pt[1]+ifr_height]]
        return IFR

    @staticmethod
    def get_ifpPolys(path, poly):
        def get_polygon_xml(polygonObj):
            polygon = []

            segments = polygonObj.getElementsByTagName("segment")
            for segment in segments:
                x0 = float(segment.getAttribute("x0"))
                y0 = float(segment.getAttribute("y0"))
                polygon.append([x0, y0])  ##逆时针添加vertex坐标

            return polygon

        ### 读取xml文件 path == '../Data-xml/fu.xml'
        xml_file = xmldom.parse(path)
        ### 获取xml文件中的元素对象
        root = xml_file.documentElement
        ## 获得子标签 return object

        polygonsObj = root.getElementsByTagName("polygon")  ## get polygon
        polygons = []  ##各种类型的polygon的定义
        dict = {}  ###借助词典进行处理

        for polygonObj in polygonsObj:
            dict[polygonObj.getAttribute("id")] = polygonObj

        ifpsObj = root.getElementsByTagName("ifp")

        ifpPolys = []
        for ifpObj in ifpsObj:
            staticPolygons = ifpObj.getElementsByTagName("staticPolygon")
            orbitingPolygons = ifpObj.getElementsByTagName("orbitingPolygon")
            resultingPolygons = ifpObj.getElementsByTagName("resultingPolygon")
            if staticPolygons[0].getAttribute("angle") == '0' and orbitingPolygons[0].getAttribute(
                    "angle") == '0':  ##只处理不旋转情况
                poly1 = get_polygon_xml(dict[staticPolygons[0].getAttribute("idPolygon")])
                poly2 = get_polygon_xml(dict[orbitingPolygons[0].getAttribute("idPolygon")])
                ifpPoly = get_polygon_xml(dict[resultingPolygons[0].getAttribute("idPolygon")])

                ifpPolys.append([poly2, ifpPoly])

        return ifpPolys

    @staticmethod
    def getInnerFitRectangle_xml(path, poly):
        ifpPolys = PackingUtil.get_ifpPolys(path, poly)
        for pair in ifpPolys:
            # print(pair[0])
            # print(poly)
            if pair[0] == poly:###这边不可以直接判别，因为poly改变过changed
                # print(pair[1])
                return pair[1]

        return -1 ## ERROR

class NFPAssistant(object):
    def __init__(self,polys,**kw):
        self.polys=PolyListProcessor.deleteRedundancy(copy.deepcopy(polys)) ##去重, 默认输入去重列表
        self.area_list,self.first_vec_list,self.centroid_list=[],[],[] # 作为参考
        self.polys_vector_list = [] ##getPolyIndex_improve参考向量
        for poly in self.polys:
            P=Polygon(poly)
            self.centroid_list.append(GeoFunc.getPt(P.centroid))
            self.area_list.append(int(P.area)) ## return  float
            self.first_vec_list.append([poly[1][0]-poly[0][0],poly[1][1]-poly[0][1]])
            poly_vector =[]
            for i in range(len(poly)):
                point1 = poly[i % len(poly)]
                point2 = poly[(i+1) % len(poly)]
                poly_vector.append([point2[0]-point1[0], point2[1]-point1[1]])
            self.polys_vector_list.append(poly_vector)

        self.nfp_list=[[0]*len(self.polys) for i in range(len(self.polys))]###矩阵存储nfp polygon
        self.load_history=False ##加载时为True
        self.history_path=None ##选择从../../record/nfp_dighe2.csv中进行加载
        self.history=None ##舍弃传入df对象，内存会爆
        self.libnfpmin_interface = True
        if 'libnfpmin_interface' in kw:
            self.libnfpmin_interface = kw["libnfpmin_interface"]

        '''
        history_path == store_path保持一致，带有name，例如fu, dighe2, ../../record/fu_nfp.csv
        ../../record/nfp.csv为默认path
        '''
        if 'history_path' in kw:
            self.history_path=kw['history_path']

        if 'load_history' in kw:
            if kw['load_history']==True:
                # 从内存中加载history 直接传递pandas的df对象 缩短I/O时间
                if 'history' in kw:
                    self.history=kw['history'] ##self.history = df对象
                self.load_history=True
                self.loadHistory()
        
        self.store_nfp=False
        if 'store_nfp' in kw:
            if kw['store_nfp']==True:
                self.store_nfp=True

        self.store_path=None ##../../path/nfp_dighe2.csv
        if 'store_path' in kw:
            self.store_path=kw['store_path']

        if self.store_nfp == True and self.store_path != None and os.path.exists(self.store_path): ##重复写入时，移除nfp文件, 然后重新写入
            os.remove(self.store_path)

        if 'get_all_nfp' in kw:
            if kw['get_all_nfp']==True and self.load_history==False:
                self.getAllNFP() ##第一次用于存储NFP

        self.xml_path = None  ##../../Data-xml/dighe2.xml
        if 'xml_path' in kw:
            self.xml_path = kw['xml_path']

        if 'get_all_nfp_xml' in kw:
            if kw['get_all_nfp_xml'] == True and self.load_history==False and self.xml_path != None:
                self.getAllNFP_xml(self.xml_path) ##第一次用于存储NFP

        if 'fast' in kw: # 为BLF进行多进程优化 ## make mistakes so ignore
            if kw['fast']==True:
                self.res=[[0]*len(self.polys) for i in range(len(self.polys))]
                # pool=Pool()
                for i in range(1,len(self.polys)):
                    for j in range(0,i):
                        # 计算nfp(j,i)
                        #self.res[j][i]=pool.apply_async(getNFP,args=(self.polys[j],self.polys[i]))
                        self.nfp_list[j][i]=GeoFunc.getSlide(getNFP(self.polys[j],self.polys[i]),-self.centroid_list[j][0],-self.centroid_list[j][1])
                # pool.close()
                # pool.join()
                # for i in range(1,len(self.polys)):
                #     for j in range(0,i):
                #         self.nfp_list[j][i]=GeoFunc.getSlide(self.res[j][i].get(),-self.centroid_list[j][0],-self.centroid_list[j][1])

    def loadHistory(self):
        if not self.history:
            if not self.history_path:
                path="../../record/nfp.csv"
            else:
                path=self.history_path
            df = pd.read_csv(path,header=None)
        else:
            df = self.history
        # df.shape[0]使用shape属性来获取DataFrame的维度信息，返回一个包含行数和列数的元组 (rows, columns)，其中索引为0的元素代表行数
        for index in range(df.shape[0]):
            # df[0]表示选取DataFrame中的第一列，df[0][index]则表示选取该列中指定索引处的值
            i=self.getPolyIndex_improve(json.loads(df[0][index]))###json.loads()它的作用是将 JSON 格式的字符串转换为 Python 对象
            j=self.getPolyIndex_improve(json.loads(df[1][index]))
            if i>=0 and j>=0:
                self.nfp_list[i][j]=json.loads(df[2][index])
        # print(self.nfp_list)

    # 获得一个形状的index 用来nfp存储
    ###如果加入旋转，此方法则同时满足面积和第一向量时返回 ###此方法有错误，有bug,且有精度问题
    def getPolyIndex(self,target):
        area=int(Polygon(target).area)
        first_vec=[target[1][0]-target[0][0],target[1][1]-target[0][1]]
        area_index=PolyListProcessor.getIndexMulti(area,self.area_list)##精度问题
        if len(area_index)==1: # 只有一个的情况
            return area_index[0]
        else:
            vec_index=PolyListProcessor.getIndexMulti(first_vec,self.first_vec_list)##精度问题
            index=[x for x in area_index if x in vec_index]
            if len(index)==0:
                print("getPolyIndex ERROR")
                import sys
                sys.exit(-1) ##ERROR
            return index[0] # 一般情况就只有一个了

    def getPolyIndex_improve(self,target): ##所有的edge vector相等，返回idx
        poly_vector = []
        for i in range(len(target)):
            point1 = target[i % len(target)]
            point2 = target[(i + 1) % len(target)]
            poly_vector.append([point2[0] - point1[0], point2[1] - point1[1]])

        for i in range(len(self.polys_vector_list)):
            if GeoFunc.vector_list_almostequal(self.polys_vector_list[i], poly_vector):
                return i

        # print("getPolyIndex_improve ERROR")
        return -1 ##表示此时的NFPassistant没有加入这个多边形，需要额外计算

    # 获得所有的形状 storeNFP
    def getAllNFP(self):
        nfp_multi=False 
        if nfp_multi==True:
            tasks=[(main,adjoin) for main in self.polys for adjoin in self.polys]
            res= Pool.starmap(NFP,tasks) ## multi processing
            for k,item in enumerate(res):
                i=k//len(self.polys)
                j=k%len(self.polys)
                self.nfp_list[i][j]=GeoFunc.getSlide(item.nfp,-self.centroid_list[i][0],-self.centroid_list[i][1])
        else:
            for i,poly1 in enumerate(self.polys):
                for j,poly2 in enumerate(self.polys):
                    ''' 默认只取最外层polygon，且保持逆时针，且封闭多边形 '''
                    if self.libnfpmin_interface == True:
                        nfp = nfp_polygon_min(poly1, poly2)[0]
                    else:
                        nfp=NFP(poly1,poly2).nfp
                    #NFP(poly1,poly2).showResult()
                    # self.nfp_list[i][j]=GeoFunc.getSlide(nfp,-self.centroid_list[i][0],-self.centroid_list[i][1])
                    self.nfp_list[i][j]=nfp
        if self.store_nfp==True:
            self.storeNFP()
    
    def storeNFP(self):
        if self.store_path==None:
            path="../../record/nfp.csv"
        else:
            path=self.store_path
        ###创建一个csv文件
        with open(path,"a+") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(self.polys)):
                for j in range(len(self.polys)):
                    writer.writerows([[self.polys[i],self.polys[j],self.nfp_list[i][j]]])

    # 输入形状获得NFP
    def getDirectNFP(self,poly1,poly2,**kw):
        i = -1
        j = -1
        if 'index' in kw:  ##ignore
            i=kw['index'][0]
            j=kw['index'][1]
            centroid=GeoFunc.getPt(Polygon(self.polys[i]).centroid)
        else:
            # 首先获得poly1和poly2的ID
            i=self.getPolyIndex_improve(poly1)
            j=self.getPolyIndex_improve(poly2)
            if i == -1 or j == -1: ##旋转情况下
                if self.libnfpmin_interface == True:
                    nfp = nfp_polygon_min (poly1, poly2)[0]
                else:
                    nfp = NFP (poly1, poly2).nfp
                return nfp
            # print(i, j)
            centroid=GeoFunc.getPt(Polygon(poly1).centroid)
        # 判断是否计算过并计算nfp
        if self.nfp_list[i][j]==0:
            ''' 默认只取最外层polygon，且保持逆时针，且封闭多边形 '''
            if self.libnfpmin_interface == True:
                nfp = nfp_polygon_min(poly1, poly2)[0]
            else:
                nfp = NFP(poly1, poly2).nfp
            #self.nfp_list[i][j]=GeoFunc.getSlide(nfp,-centroid[0],-centroid[1])
            if self.store_nfp==True:
                if self.store_path == None:
                    path = "../../record/nfp.csv"
                else:
                    path = self.store_path
                with open(path,"a+") as csvfile:
                    writer = csv.writer(csvfile)
                    # self.nfp_list[i][j]=GeoFunc.getSlide(nfp, -centroid[0], -centroid[1]) ##mark
                    writer.writerows([[poly1,poly2,nfp]])
            return nfp
        else: ###默认进行这一步
            # return GeoFunc.getSlide(self.nfp_list[i][j],centroid[0],centroid[1])
            return self.nfp_list[i][j]

    def get_polygon_xml(self, polygonObj):
        polygon = []

        segments = polygonObj.getElementsByTagName("segment")
        for segment in segments:
            x0 = float(segment.getAttribute("x0"))
            y0 = float(segment.getAttribute("y0"))
            polygon.append([x0, y0])  ##逆时针添加vertex坐标

        return polygon

    def getAllNFP_xml(self, path):
        ### 读取xml文件 path == '../Data-xml/fu.xml'
        xml_file = xmldom.parse(path)
        ### 获取xml文件中的元素对象
        root = xml_file.documentElement
        ## 获得子标签 return object

        polygonsObj = root.getElementsByTagName("polygon")  ## get polygon
        polygons = []  ##各种类型的polygon的定义
        dict = {}  ###借助词典进行处理

        for polygonObj in polygonsObj:
            dict[polygonObj.getAttribute("id")] = polygonObj

        nfpsObj = root.getElementsByTagName("nfp")

        for nfpObj in nfpsObj:
            staticPolygons = nfpObj.getElementsByTagName("staticPolygon")
            orbitingPolygons = nfpObj.getElementsByTagName("orbitingPolygon")
            resultingPolygons = nfpObj.getElementsByTagName("resultingPolygon")
            if staticPolygons[0].getAttribute("angle") == '0' and orbitingPolygons[0].getAttribute(
                    "angle") == '0':  ##只处理不旋转情况
                poly1 = self.get_polygon_xml(dict[staticPolygons[0].getAttribute("idPolygon")])
                poly2 = self.get_polygon_xml(dict[orbitingPolygons[0].getAttribute("idPolygon")])
                nfpPoly = self.get_polygon_xml(dict[resultingPolygons[0].getAttribute("idPolygon")])

                # start_point_index = GeoFunc.checkBottom(poly1)  ## static polygon bottom point
                # start_point=[poly1[start_point_index][0],poly1[start_point_index][1]]
                #
                # nfp_start_point_index = GeoFunc.checkBottom(nfpPoly)  ## static polygon bottom point
                # nfp_start_point = [nfpPoly[nfp_start_point_index][0], nfpPoly[nfp_start_point_index][1]]
                #
                # GeoFunc.slideToPoint(nfpPoly, nfp_start_point, start_point)###规范化, bottom point start point 与NFP算法保持一致

                # centroid = GeoFunc.getPt(Polygon(poly1).centroid)
                # GeoFunc.slidePoly(nfpPoly, -centroid[0], -centroid[1])
                if self.store_nfp == True:
                    if self.store_path == None:
                        path = "../../record/nfp.csv"
                    else:
                        path = self.store_path
                    ###创建一个csv文件
                    with open(path, "a+") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([[poly1, poly2, nfpPoly]])

class PolyListProcessor(object):
    @staticmethod
    def getPolyObjectList(polys,allowed_rotation):
        '''
        将Polys和允许旋转的角度转化为poly_lists
        '''
        poly_list=[]
        for i,poly in enumerate(polys): ##enumerate(iterable, start=0)
            poly_list.append(Poly(i,poly,allowed_rotation))
        return poly_list

    @staticmethod
    def getPolysVertices(_list):
        '''排序结束后会影响'''
        polys=[]
        for i in range(len(_list)):
            polys.append(_list[i].poly)
        return polys
    
    @staticmethod
    def getPolysVerticesCopy(_list):
        '''不影响list内的形状'''
        polys=[]
        for i in range(len(_list)):
            polys.append(copy.deepcopy(_list[i].poly))
        return polys

    @staticmethod
    def getPolyListIndex(poly_list):
        index_list=[]
        for i in range(len(poly_list)):
            index_list.append(poly_list[i].num)
        return index_list
    
    @staticmethod
    def getIndex(item,_list):
        for i in range(len(_list)):
            if item==_list[i]:
                return i
        return -1
    
    @staticmethod
    def getIndexMulti(item,_list):
        index_list=[]
        for i in range(len(_list)):
            if item==_list[i]:
                index_list.append(i)
        return index_list

    @staticmethod
    def randomSwap(poly_list,target_id):
        new_poly_list=copy.deepcopy(poly_list)

        swap_with = int(random.random() * len(new_poly_list))
        
        item1 = new_poly_list[target_id]
        item2 = new_poly_list[swap_with]
            
        new_poly_list[target_id] = item2
        new_poly_list[swap_with] = item1
        return new_poly_list

    @staticmethod
    def randomRotate(poly_list,min_angle,target_id):
        new_poly_list=copy.deepcopy(poly_list)

        index = random.randint(0,len(new_poly_list)-1)###[]闭区间
        RatotionPoly(min_angle).rotation(new_poly_list[index].poly)
        return new_poly_list

    @staticmethod
    def showPolyList(width,poly_list):
        blf = BLF.BottomLeftFill(width,PolyListProcessor.getPolysVertices(poly_list))
        blf.showAll()

    @staticmethod
    def deleteRedundancy(_arr):
        new_arr = []
        for item in _arr:
            if not item in new_arr:
                new_arr.append(item)
        return new_arr

    @staticmethod
    def getPolysByIndex(index_list,poly_list):
        choosed_poly_list=[]
        for i in index_list:
            choosed_poly_list.append(poly_list[i])
        return choosed_poly_list

class RatotionPoly():
    def __init__(self,angle):
        self.angle=angle
        self._max=360/angle

    def rotation(self,poly):
        if self._max>1:
            # print("旋转图形")
            rotation_res=random.randint(1,self._max-1) ###[]闭区间
            Poly=Polygon(poly)
            new_Poly=affinity.rotate(Poly,rotation_res*self.angle)
            mapping_res=mapping(new_Poly)
            new_poly=mapping_res["coordinates"][0]
            for index in range(0,len(poly)):
                poly[index]=[new_poly[index][0],new_poly[index][1]]
        else:
            pass
            # print("不允许旋转")

    def rotation_specific(self,poly,angle=-1):
        '''
        旋转特定角度
        '''
        Poly=Polygon(poly)
        if angle==-1: angle=self.angle
        elif len(angle)>0:
            angle=np.random.choice(angle)
            # print('旋转{}°'.format(angle))
        new_Poly=affinity.rotate(Poly,angle)
        mapping_res=mapping(new_Poly)
        new_poly=mapping_res["coordinates"][0]
        for index in range(0,len(poly)):
            poly[index]=[new_poly[index][0],new_poly[index][1]]

if __name__=='__main__':
    ### test code
    # poly = [[0, 0], [2, 0.5], [1.5, 1], [0.5, 1]] ## polygon逆时针
    # ifr = PackingUtil.getInnerFitRectangle(poly, 10, 5)
    # PltFunc.addPolygonColor(poly)
    # PltFunc.addPolygon(ifr)
    # plt.axis([0, 10, 0, 10])
    # plt.show()

    path = '../../Data-xml/swim.xml'
    polys_DR = process_data_xml_deleteRedundancy(path)
    polys = process_data_xml(path)
    width = 5752
    # polys_idx = [0, 1, 2, 3, 4, 6, 5, 7, 8, 9]
    polys_idx = [i for i in range(len(polys))]
    tmp_polys = [polys[i] for i in polys_idx]
    ###第一次将NFP写入当地文件. 默认输入去重之后的polys
    NFPAssistant(polys_DR, store_nfp=True, store_path='../../record/swim_nfp.csv', get_all_nfp=True)
    # NFPAssistant(polys_DR, store_nfp=True, store_path='../../record/swim_nfp.csv', get_all_nfp_xml=True, xml_path=path)
    ###后序直接获取文件，进行处理
    nfp_ass = NFPAssistant(polys_DR, load_history=True, history_path='../../record/swim_nfp.csv')
    bfl = BLF.BottomLeftFill(width, tmp_polys, vertical=True, NFPAssistant=nfp_ass)
    bfl.textPrint()
    bfl.showAll(usage=bfl.patches_area/bfl.board_area)
    # bfl.showAll()
from shapely.geometry import Polygon,Point,mapping,LineString
import math
import numpy as np


bias=0.000001 # 计算精度偏差

class GeoFunc(object):
    '''
    几何相关函数 此类没有加self，所以函数都是static
    1. checkBottom、checkTop、checkLeft、checkRight暂时不考虑多个点
    2. checkBottom和checkLeft均考虑最左下角## 方法需要改进
    3. checkTop和checkRight考虑最右上角
    '''
    def almostContain(line,point):
        # 会由于int导致计算偏差！！！！！！
        pt1=[line[0][0],line[0][1]]
        pt2=[line[1][0],line[1][1]]
        point=[point[0],point[1]]

        # 水平直线情况：通过比较两个点和中间点比较
        if abs(pt1[1]-point[1])<bias and abs(pt2[1]-point[1])<bias:
            # print("水平情况")
            if (pt1[0]-point[0])*(pt2[0]-point[0])<0:
                return True
            else:
                return False
    
        # 排除垂直的情况
        if abs(pt1[0]-point[0])<bias and abs(pt2[0]-point[0])<bias:
            # print("垂直情况")
            if (pt1[1]-point[1])*(pt2[1]-point[1])<0:
                return True
            else:
                return False

        if abs(pt1[0]-point[0])<bias or abs(pt2[0]-point[0])<bias or abs(pt1[0]-pt2[0])<bias:
            return False
        
        # 正常情况，计算弧度的差值
        arc1=np.arctan((line[0][1]-line[1][1])/(line[0][0]-line[1][0]))
        arc2=np.arctan((point[1]-line[1][1])/(point[0]-line[1][0]))
        if abs(arc1-arc2)<bias: # 原值0.03，dighe近似平行修正为0.01
            if (point[1]-pt1[1])*(pt2[1]-point[1])>0 and (point[0]-pt1[0])*(pt2[0]-point[0])>0:
                # print("一般情况")
                return True
            else:
                return False
        else:
            return False
    
    def computeInterArea(orginal_inter):
        '''
        计算相交区域的面积
        '''
        inter=mapping(orginal_inter)
        # 一个多边形
        if inter["type"]=="Polygon":
            if len(inter["coordinates"])>0:
                poly=inter["coordinates"][0]
                return Polygon(poly).area
            else: return 0
        if inter["type"]=="MultiPolygon":
            area=0
            for _arr in inter["coordinates"]:
                poly=_arr[0]
                area=area+Polygon(poly).area
            return area
        
        if inter["type"]=="GeometryCollection":
            area=0
            for _arr in inter["geometries"]:
                if _arr["type"]=="Polygon":
                    poly=_arr["coordinates"][0]
                    area=area+Polygon(poly).area
            return area
        return 0

    def checkBottom(poly): ## re
        # turn the bottom-left point index
        polyP=Polygon(poly)
        min_y=polyP.bounds[1] ## Returns minimum bounding region (minx, miny, maxx, maxy)
        index_list = []
        for index,point in enumerate(poly):###寻求最下方的索引列表
            if point[1]==min_y:
                index_list.append(index)

        if len(index_list) == 1:
            return index_list[0]

        tmp = poly[index_list[0]][0] ##拿到一个横坐标
        res = index_list[0] ## index
        for i in range(1, len(index_list)):###得到最左边的索引
            if poly[index_list[i]][0] < tmp:
                tmp = poly[index_list[i]][0]
                res = index_list[i]

        return res

    def checkTop(poly):
        polyP=Polygon(poly)
        max_y=polyP.bounds[3]
        index_list = []
        for index,point in enumerate(poly):###寻求最上边的索引列表
            if point[1]==max_y:
                index_list.append(index)

        if len(index_list) == 1:
            return index_list[0]

        tmp = poly[index_list[0]][0]  ##拿到一个横坐标
        res = index_list[0]  ## index
        for i in range(1, len(index_list)):  ###得到最右边的索引
            if poly[index_list[i]][0] > tmp:
                tmp = poly[index_list[i]][0]
                res = index_list[i]

        return res

    def checkLeft(poly):
        polyP=Polygon(poly)
        min_x=polyP.bounds[0]
        index_list = []
        for index,point in enumerate(poly):###寻求最左边的索引列表
            if point[0]==min_x:
                index_list.append(index)

        if len(index_list) == 1:
            return index_list[0]

        tmp = poly[index_list[0]][1]  ##拿到一个纵坐标
        res = index_list[0]  ## index
        for i in range(1, len(index_list)):  ###得到最下边的索引
            if poly[index_list[i]][1] < tmp:
                tmp = poly[index_list[i]][1]
                res = index_list[i]

        return res

    def checkRight(poly):
        polyP=Polygon(poly)
        max_x=polyP.bounds[2]
        index_list = []
        for index,point in enumerate(poly):###寻求最右边的索引列表
            if point[0]==max_x:
                index_list.append(index)

        if len(index_list) == 1:
            return index_list[0]

        tmp = poly[index_list[0]][1]  ##拿到一个纵坐标
        res = index_list[0]  ## index
        for i in range(1, len(index_list)):  ###得到最上边的索引
            if poly[index_list[i]][1] < tmp:
                tmp = poly[index_list[i]][1]
                res = index_list[i]

        return res

    def checkBound(poly):
        return GeoFunc.checkLeft(poly), GeoFunc.checkBottom(poly), GeoFunc.checkRight(poly), GeoFunc.checkTop(poly)

    # Create a square polygon with no holes
    # >> > coords = ((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.))
    # >> > polygon = Polygon(coords)
    # >> > polygon.area
    # 1.0

    def checkBoundPt(poly):
        '''获得边界的点'''
        left,bottom,right,top=poly[0],poly[0],poly[0],poly[0]
        for i,pt in enumerate(poly):
            if pt[0]<left[0]:
                left=pt
            if pt[0]>right[0]:
                right=pt
            if pt[1]>top[1]:
                top=pt
            if pt[1]<bottom[1]:
                bottom=pt
        return left,bottom,right,top

    def checkBoundValue(poly):
        '''获得边界的值即xy值'''
        left,bottom,right,top=poly[0][0],poly[0][1],poly[0][0],poly[0][1]
        for i,pt in enumerate(poly):
            if pt[0]<left:
                left=pt[0]
            if pt[0]>right:
                right=pt[0]
            if pt[1]>top:
                top=pt[1]
            if pt[1]<bottom:
                bottom=pt[1]
        return left,bottom,right,top

    def slideToPoint(poly,pt1,pt2):
        ##将poly多边形中点pt1移动到点pt2
        GeoFunc.slidePoly(poly,pt2[0]-pt1[0],pt2[1]-pt1[1])

    # return moved polygon
    def getSlideToPoint(poly, pt1, pt2):
        return GeoFunc.getSlide(poly,pt2[0]-pt1[0],pt2[1]-pt1[1])

    def getSlide(poly,x,y):
        '''
        获得平移后的情况，return list == 功能等同于slidePoly, no return
        '''
        new_vertex=[]
        for point in poly:
            new_point=[point[0]+x,point[1]+y]
            new_vertex.append(new_point)
        return new_vertex

    @staticmethod
    def boundsContain(bounds, pt):
        if pt[0] >= bounds[0] and pt[0] <= bounds[2] and pt[1] >= bounds[1] and pt[1] <= bounds[3]:
            return True
        return False

    def slidePoly(poly,x,y):
        for point in poly:
            point[0]=point[0]+x
            point[1]=point[1]+y

    def polyToArr(inter):
        res=mapping(inter)
        _arr=[]
        if res["type"]=="MultiPolygon":
            for poly in res["coordinates"]:
                for point in poly[0]:
                    _arr.append([point[0],point[1]])
        elif res["type"]=="GeometryCollection":
            for item in res["geometries"]:
                if item["type"]=="Polygon":
                    for point in item["coordinates"][0]:
                        _arr.append([point[0],point[1]])
        else:
            if res["coordinates"][0][0]==res["coordinates"][0][-1]:
                for point in res["coordinates"][0][0:-1]:
                    _arr.append([point[0],point[1]])
            else:
                for point in res["coordinates"][0]:
                    _arr.append([point[0],point[1]])
        return _arr

    def normData(poly,num):
        for ver in poly:
            ver[0]=ver[0]*num
            ver[1]=ver[1]*num

    '''近似计算'''
    def crossProduct(vec1,vec2):
        res=vec1[0]*vec2[1]-vec1[1]*vec2[0]
        # 最简单的计算
        if abs(res)<bias:
            return 0
        # 部分情况叉积很大但是仍然基本平行
        if abs(vec1[0])>bias and abs(vec2[0])>bias:
            if abs(vec1[1]/vec1[0]-vec2[1]/vec2[0])<bias:
                return 0
        return res

    '''用于touching计算交点 可以与另一个交点计算函数合并'''
    def intersection(line1,line2):
        # 如果可以直接计算出交点
        Line1=LineString(line1)
        Line2=LineString(line2)
        inter=Line1.intersection(Line2)
        if inter.is_empty==False:
            mapping_inter=mapping(inter)
            if mapping_inter["type"]=="LineString": ##重合 是否直接返回整条lineString的首尾
                inter_coor=mapping_inter["coordinates"][0]
                # start_pt= mapping_inter["coordinates"][0] ##(0, 0)
                # end_pt = mapping_inter["coordinates"][1]
                # inter_coor=[[start_pt[0], start_pt[1]], [end_pt[0], end_pt[1]]]
            else:#mapping_inter["type"]=="Point":
                inter_coor=mapping_inter["coordinates"]
            return [inter_coor[0], inter_coor[1]] ##inter_coor == (0, 0) ---> [inter_coor[0], inter_coor[1]] changed

        # 对照所有顶点是否相同
        res=[]
        for pt1 in line1:
            for pt2 in line2:
                if GeoFunc.almostEqual(pt1,pt2)==True:
                    # print("pt1,pt2:",pt1,pt2)
                    res=pt1
        if res!=[]:
            return res

        # 计算是否存在almostContain
        for pt in line1:
            if GeoFunc.almostContain(line2,pt)==True:
                return pt
        for pt in line2:
            if GeoFunc.almostContain(line1,pt)==True:
                return pt
        return []

    ''' 主要用于判断是否有直线重合'''
    def newLineInter(line1,line2):
        vec1=GeoFunc.lineToVec(line1)
        vec2=GeoFunc.lineToVec(line2)
        vec12_product=GeoFunc.crossProduct(vec1,vec2)
        Line1=LineString(line1)
        Line2=LineString(line2)
        inter={
            "length":0, ##return重合的线长度
            "geom_type":None
        }
        # 只有平行才会有直线重叠
        if vec12_product==0:
            # copy避免影响原值
            new_line1=GeoFunc.copyPoly(line1)
            new_line2=GeoFunc.copyPoly(line2)
            ##逆向量即向量乘以-1
            # if vec1[0] * vec2[0] < 0 and vec1[1] * vec2[1] < 0:
            if vec1[0]*vec2[0]<0 or vec1[1]*vec2[1]<0:
                new_line2=GeoFunc.reverseLine(new_line2)
            # 如果存在顶点相等，则选择其中一个
            if GeoFunc.almostEqual(new_line1[0],new_line2[0]) or GeoFunc.almostEqual(new_line1[1],new_line2[1]):
                inter["length"]=min(Line1.length,Line2.length)
                inter["geom_type"]='LineString'
                return inter
            # 排除只有顶点相交情况
            if GeoFunc.almostEqual(new_line1[0],new_line2[1]):
                # inter["length"]=new_line2[1]
                inter["geom_type"]='Point'
                return inter
            if GeoFunc.almostEqual(new_line1[1],new_line2[0]):
                # inter["length"]=new_line1[1]
                inter["geom_type"]='Point'
                return inter
            # 否则判断是否包含
            line1_contain_line2_pt0=GeoFunc.almostContain(new_line1,new_line2[0])
            line1_contain_line2_pt1=GeoFunc.almostContain(new_line1,new_line2[1])
            line2_contain_line1_pt0=GeoFunc.almostContain(new_line2,new_line1[0])
            line2_contain_line1_pt1=GeoFunc.almostContain(new_line2,new_line1[1])
            # Line1直接包含Line2
            if line1_contain_line2_pt0 and line1_contain_line2_pt1:
                inter["length"]=Line1.length
                inter["geom_type"]='LineString'
                return inter
            # Line2直接包含Line1
            if line1_contain_line2_pt0 and line1_contain_line2_pt1:
                inter["length"]=Line2.length
                inter["geom_type"]='LineString'
                return inter
            # 相互包含交点
            if line1_contain_line2_pt0 and line2_contain_line1_pt1:
                inter["length"]=LineString([line2[0],line1[1]]).length
                inter["geom_type"]='LineString'
                return inter
            if line1_contain_line2_pt1 and line2_contain_line1_pt0:
                inter["length"]=LineString([line2[1],line1[0]]).length
                inter["geom_type"]='LineString'
                return inter
        return inter

    def newLineInter_simplified(line1, line2):
        Line1 = LineString(line1)
        Line2 = LineString(line2)
        inter = Line1.intersection(Line2)
        strs = None
        length = 0.
        if inter.is_empty == False:
            mapping_inter = mapping(inter)
            if mapping_inter["type"] == "LineString":  ##重合 是否直接返回整条lineString的首尾
                strs = "LineString"
                length = LineString(mapping_inter["coordinates"]).length
            else:  # mapping_inter["type"]=="Point":
                strs= "Point"
                length = 0.

        # if inter.is_empty == False:
        return {"length":length, "geom_type": strs}##return重合的 line_length and geometry type

        # return GeoFunc.newLineInter(line1, line2)
    def reverseLine(line):
        pt0=line[0]
        pt1=line[1]
        return [[pt1[0],pt1[1]],[pt0[0],pt0[1]]]

    '''近似计算'''
    def almostEqual(point1,point2):
        if abs(point1[0]-point2[0])<bias and abs(point1[1]-point2[1])<bias:
            return True
        else:
            return False

    ##通过直接判断edge_vector_list相等, 间接判断polygon是否相等，由于polygon移动，所以有此方法[[],[]]
    def vector_list_almostequal(poly, target):  ### list 数组
        if len(poly) != len(target): return False
        n = len(poly)
        for i in range(n):
            point1 = poly[i]
            point2 = target[i]
            if not GeoFunc.almostEqual(point1, point2):##精度问题
                return False
        return True

    def extendLine(line):
        '''
        直线延长
        '''
        pt0=line[0]
        pt1=line[1]
        vect01=[pt1[0]-pt0[0],pt1[1]-pt0[1]]
        vect10=[-vect01[0],-vect01[1]]
        multi=40
        new_pt1=[pt0[0]+vect01[0]*multi,pt0[1]+vect01[1]*multi]
        new_pt0=[pt1[0]+vect10[0]*multi,pt1[1]+vect10[1]*multi]
        return [new_pt0,new_pt1]

    def getArc(line):
        if abs(line[0][0]-line[1][0])<0.01: # 垂直情况
            if line[0][1]-line[1][1]>0:
                return 0.5*math.pi
            else:
                return -0.5*math.pi
        k=(line[0][1]-line[1][1])/(line[0][0]-line[1][0])
        arc=np.arctan(k)
        return arc

    def extendInter(line1,line2):
        '''
        获得延长线的交点
        '''
        line1_extend=GeoFunc.extendLine(line1)
        line2_extend=GeoFunc.extendLine(line2)
        # 排查平行情况
        k1=GeoFunc.getArc(line1_extend)
        k2=GeoFunc.getArc(line2_extend)
        if abs(k1-k2)<0.01:
            return [line1[1][0],line1[1][1]]
        inter=mapping(LineString(line1_extend).intersection(LineString(line2_extend)))
        if inter["type"]=="GeometryCollection" or inter["type"]=="LineString":
            return [line1[1][0],line1[1][1]]
        return [inter["coordinates"][0],inter["coordinates"][1]]

    def twoDec(poly):
        for pt in poly:
            pt[0]=round(pt[0],2)
            pt[1]=round(pt[1],2)

    def similarPoly(poly):
        '''
        求解凸多边形的近似多边形，凹多边形内凹部分额外处理
        '''
        change_len=10
        extend_poly=poly+poly
        Poly=Polygon(poly)
        new_edges=[]
        # 计算直线平移
        for i in range(len(poly)):
            line=[extend_poly[i],extend_poly[i+1]]
            new_line=GeoFunc.slideOutLine(line,Poly,change_len)
            new_edges.append(new_line)

        # 计算直线延长线
        new_poly=[]
        new_edges.append(new_edges[0])
        for i in range(len(new_edges)-1):
            inter=GeoFunc.extendInter(new_edges[i],new_edges[i+1])
            new_poly.append(inter)

        GeoFunc.twoDec(new_poly)

        return new_poly

    def slideOutLine(line,Poly,change_len):
        '''
        向外平移直线
        '''
        pt0=line[0]
        pt1=line[1]
        mid=[(pt0[0]+pt1[0])/2,(pt0[1]+pt1[1])/2]
        if pt0[1]!=pt1[1]:
            k=-(pt0[0]-pt1[0])/(pt0[1]-pt1[1]) # 垂直直线情况
            theta=math.atan(k)
            delta_x=1*math.cos(theta)
            delta_y=1*math.sin(theta)
            if Poly.contains(Point([mid[0]+delta_x,mid[1]+delta_y])):
                delta_x=-delta_x
                delta_y=-delta_y
            new_line=[[pt0[0]+change_len*delta_x,pt0[1]+change_len*delta_y],[pt1[0]+change_len*delta_x,pt1[1]+change_len*delta_y]]
            return new_line
        else:
            delta_y=1
            if Poly.contains(Point([mid[0],mid[1]+delta_y])):
                delta_y=-delta_y
            return [[pt0[0],pt0[1]+change_len*delta_y],[pt1[0],pt1[1]+change_len*delta_y]]

    def copyPoly(poly):
        new_poly=[]
        for pt in poly:
            new_poly.append([pt[0],pt[1]])
        return new_poly

    def pointLineDistance(point, line):
        point_x = point[0]
        point_y = point[1]
        line_s_x = line[0][0]
        line_s_y = line[0][1]
        line_e_x = line[1][0]
        line_e_y = line[1][1]
        if line_e_x - line_s_x == 0:
            return abs(point_x - line_s_x),[line_s_x-point_x,0]
        if line_e_y - line_s_y == 0:
            return abs(point_y - line_s_y),[0,line_s_y-point_y]

        k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
        extend_line=[[point_x-1000,point_y-1000*(-1/k)],[point_x+1000,point_y+1000*(-1/k)]]
        inter=LineString(line).intersection(LineString(extend_line))
        if inter.is_empty==True:
            dis1=math.pow((point_x-line_s_x)*(point_x-line_s_x)+(point_y-line_s_y)*(point_y-line_s_y), 0.5)
            dis2=math.pow((point_x-line_e_x)*(point_x-line_e_x)+(point_y-line_e_y)*(point_y-line_e_y), 0.5)
            if dis1>dis2:
                return dis2,[line_e_x-point_x,line_e_y-point_y]
            else:
                return dis1,[line_s_x-point_x,line_s_y-point_y]
        else:
            pt=GeoFunc.getPt(inter)
            dis=math.pow((point_x-pt[0])*(point_x-pt[0])+(point_y-pt[1])*(point_y-pt[1]), 0.5)
            return dis,[pt[0]-point[0],pt[1]-point[1]]

    def getPt(point):
        mapping_result=mapping(point)
        return [mapping_result["coordinates"][0],mapping_result["coordinates"][1]]

    # 获得某个多边形的边
    def getPolyEdges(poly):
        edges=[]
        for index,point in enumerate(poly):
            if index < len(poly)-1:
                edges.append([poly[index],poly[index+1]])
            else:
                edges.append([poly[index],poly[0]])
        return edges

    def pointPrecisionChange(pt,num):
        return [round(pt[0],num),round(pt[1],num)]

    def linePrecisionChange(line,num):
        return [GeoFunc.pointPrecisionChange(line[0],num),GeoFunc.pointPrecisionChange(line[1],num)]

    def lineToVec(edge):
        return [edge[1][0]-edge[0][0],edge[1][1]-edge[0][1]]

    '''可能需要用近似计算进行封装！！！！！！'''
    def judgePosition(edge1,edge2): ##用坐标叉积进行判别，将向量移动到同一起点
        x1=edge1[1][0]-edge1[0][0]
        y1=edge1[1][1]-edge1[0][1]
        x2=edge2[1][0]-edge2[0][0]
        y2=edge2[1][1]-edge2[0][1]
        res=x1*y2-x2*y1
        right=False
        left=False
        parallel=False
        # print("res:",res)
        if res==0:
            parallel=True
        elif res>0:
            left=True ##edge2在edge1的左边, 顺时针
        else:
            right=True ##edge2在edge1的右边，逆时针
        return right,left,parallel


    def getSlideLine(line,x,y):
        new_line=[]
        for pt in line:
            new_line.append([pt[0]+x,pt[1]+y])
        return new_line

    def getCentroid(poly):
        return GeoFunc.getPt(Polygon(poly).centroid)

if __name__ == "__main__":
    # out = GeoFunc.newLineInter_simplified([[0, 0], [1, 1]], [[1, 1], [2, 2]])
    # print(out)
    pass
import pandas as pd ##xlrd同样也可以读取xls文件
import xml.dom.minidom as xmldom

# return the polygons list and width
def process_data_xls(path, sheet_name=None):
    if sheet_name == None:
        dighe1 = pd.read_excel(path)
    else:
        dighe1 = pd.read_excel(path, sheet_name=sheet_name)

    polygons = []
    ##读取polygon坐标
    for i in range(4, dighe1.shape[0], 2):  ###行 X Y
        poly = []
        for j in range(2, dighe1.shape[1]):  ###列
            if not pd.isna(dighe1.iloc[i, j]):
                # print([dighe1.iloc[i, j], dighe1.iloc[i+1, j]])
                poly.append([dighe1.iloc[i, j], dighe1.iloc[i + 1, j]])
        # print(poly)
        if len(poly) != 0:
            polygons.append(poly)

    width = 0.
    for i in range(3, dighe1.shape[1]):
        if not pd.isna(dighe1.iloc[2, i]):
            width = dighe1.iloc[2, i]
            break

    return (polygons, width)

## return the polygons list 去重(piece)
def process_data_xml_deleteRedundancy(path):
    ### 读取xml文件 path == '../Data-xml/fu.xml'
    xml_file = xmldom.parse(path)
    ### 获取xml文件中的元素对象
    root = xml_file.documentElement
    ## 获得子标签 return object

    polygonsObj = root.getElementsByTagName("polygon")  ## get polygon
    polygons = [] ##各种类型的polygon的定义
    for polygonObj in polygonsObj:
        if polygonObj.hasAttribute("id"):
            strs = polygonObj.getAttribute("id")
            if strs != "polygon0" and strs[0:7] == "polygon": ###去除第一个无用polygon
                polygon = []
                # print(strs)
                # xmax = polygonObj.getElementsByTagName("xMax")[0]  ## node object
                # print(xmax.nodeName, ":", xmax.childNodes[0].data)
                segments = polygonObj.getElementsByTagName("segment")
                for segment in segments:
                    x0 = float(segment.getAttribute("x0"))
                    y0 = float(segment.getAttribute("y0"))
                    # x1 = float(segment.getAttribute("x1"))
                    # y1 = float(segment.getAttribute("y1"))
                    polygon.append([x0, y0]) ##逆时针添加vertex坐标
                polygons.append(polygon)

    return polygons

def process_data_xml(path):
    polygons = process_data_xml_deleteRedundancy(path)
    ### 读取xml文件 path == '../Data-xml/fu.xml'
    xml_file = xmldom.parse(path)
    ### 获取xml文件中的元素对象
    root = xml_file.documentElement
    ## 获得子标签 return object

    quantitysObj = root.getElementsByTagName("piece")  ## get quantitysObj
    quantitys = [] ###各种类型的polygon数量

    for quantityObj in quantitysObj:
        ID = quantityObj.getAttribute("id")
        if ID[0:5] == "piece":
            quantitys.append(int(quantityObj.getAttribute("quantity")))
    # print(quantitys)

    result = [] ##综合polygons和quantitys
    for i in range(len(quantitys)):
        for j in range(quantitys[i]):
            result.append(polygons[i])

    return result
if __name__ == "__main__":
    # path = "../../Data-xls/dighe.xls"
    # polygons = process_data_xls(path, "Dighe2")[0]
    # size = len(polygons)
    # width = process_data_xls(path, "Dighe2")[1]
    # print(polygons)
    # print(size)
    # print(width)

    path = '../../Data-xml/trousers.xml'
    width = 100
    polys_DR = process_data_xml_deleteRedundancy(path)
    polys = process_data_xml(path)
    print(len(polys_DR), len(polys))
    # packing.NFPAssistant(polys_DR, store_nfp=True, store_path='../../record/dighe1_nfp.csv', get_all_nfp_xml=True,
    #              xml_path=path)
    # nfp_ass = packing.NFPAssistant(polys_DR, load_history=True, history_path='../../record/dighe1_nfp.csv')
    # for i, poly1 in enumerate(polys):
    #     for j, poly2 in enumerate(polys):
    #         nfp = nfp_ass.getDirectNFP(poly1, poly2)
    #         print(i, j)
    #         PltFunc.addPolygon(poly1)
    #
    #         start_point_index = GeoFunc.checkBottom(poly1)  ## bottom point
    #         start_point = [poly1[start_point_index][0], poly1[start_point_index][1]]
    #         locus_index = GeoFunc.checkTop(poly2)  ## locus index 轨迹索引 top point 通过索引方便拿到移动的之后的点坐标
    #         # 如果不加list则original_top是指针
    #         original_top = list(poly2[locus_index])
    #         GeoFunc.slideToPoint(poly2, original_top, start_point)
    #         PltFunc.addPolygon(poly2)
    #
    #         ##nfp移动到main的bottom处
    #         nfp_start_point_index = GeoFunc.checkBottom(nfp)  ## static polygon bottom point
    #         nfp_start_point = [nfp[nfp_start_point_index][0], nfp[nfp_start_point_index][1]]
    #         GeoFunc.slideToPoint(nfp, nfp_start_point, start_point)  ###规范化, bottom point start point 与NFP算法保持一致
    #         PltFunc.addPolygonColor(nfp)
    #         PltFunc.showPlt(width=100, height=100)



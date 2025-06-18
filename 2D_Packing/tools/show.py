import matplotlib.pyplot as plt

class PltFunc(object):

    def addPolygon(poly):
        for i in range(0,len(poly)):
            if i == len(poly)-1:
                PltFunc.addLine([poly[i],poly[0]])
            else:
                PltFunc.addLine([poly[i],poly[i+1]])

    def addPolygonColor(poly): ### 通过逐个给线染色，add polygon color
        for i in range(0,len(poly)):
            if i == len(poly)-1:
                PltFunc.addLine([poly[i],poly[0]],color="blue",)
            else:
                PltFunc.addLine([poly[i],poly[i+1]],color="blue")

    def addLine(line,**kw):
        if len(kw)==0:
            plt.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],color="black",linewidth=0.5)
        else:
            plt.plot([line[0][0],line[1][0]],[line[0][1],line[1][1]],color=kw["color"],linewidth=2)

    def addPolygon_fill(poly):
        X = [i[0] for i in poly] ##横纵坐标列表
        Y = [i[1] for i in poly]

        plt.fill(X, Y, color='green') ##填充
        for i in range(len(poly)): ##线框染色
            if i == len(poly)-1:
                PltFunc.addLine([poly[i],poly[0]])
            else:
                PltFunc.addLine([poly[i],poly[i+1]])

    def showPlt(*args, **kw): ## kw 字典,解析文件成为字典，然后通过列出 figure 通过plt.show()

        if len(kw)>0:
            if "minus" in kw:
                plt.axhline(y=0,c="blue") ##horizon
                plt.axvline(x=0,c="blue") ##vertical
                plt.axis([-kw["minus"],kw["width"],-kw["minus"],kw["height"]])
                
            else:
                plt.axis([0,kw["width"],0,kw["height"]]) ###默认
        else:
            plt.axis([0,100,0,100])

        if 'usage' in kw:## title
            plt.title("usage: " + str(kw['usage']))
        if 'fig_path' in kw : ##先存储后show
            if 'only_save' in args: ###直接return
                # import os
                # if os.path.exists(kw['fig_path']):
                #     os.remove(kw['fig_path'])
                plt.gca().set_aspect(1)
                plt.savefig(kw['fig_path'], dpi=1000)  ##dpi设置1000，更清晰
                plt.clf()
                return
            else:
                plt.gca().set_aspect(1)
                plt.savefig(kw['fig_path'], dpi=1000)  ##dpi设置1000，更清晰


        # import matplotlib
        # matplotlib.use('TkAgg')  ##debug 显示图片
        # plt.show(block=True) ##debug block 显示图片
        # plt.figure(dpi=1000)  # 设置整个 figure 的 DPI
        # plt.margins(0)  # 取消默认边距
        # plt.tight_layout()  # 让内容填充整个图像
        plt.gca().set_aspect(1) ## 让 x 轴和 y 轴单位长度相等
        plt.show()
        plt.clf() #"""Clear the current figure."""

    def showPolys(polys):
        for poly in polys:
            PltFunc.addPolygon(poly)
        PltFunc.showPlt(width=2000,height=2000)
    
if __name__ == "__main__":
    p = PltFunc()
    p.addPolygon()
#https://www.jb51.net/article/164018.htm
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# data1_loss =np.loadtxt("D:\HZQ\jtrafficsigns\yolov3-point\训练的数据\未作任何改变\未作任何改变的results.txt")  # 原始
data2_loss = np.loadtxt("D:\HZQ\jtrafficsigns\yolov3-point\yolov3-tiny-3l-9a-cbam-hzq-results.txt")  # kmeans++
data3_loss = np.loadtxt("D:\HZQ\jtrafficsigns\yolov3-point\yolov3-tiny-3l-9a-results.txt")  # add-CBAM
data4_loss = np.loadtxt("D:\HZQ\jtrafficsigns\yolov3-point\yolov3-tiny-cbam_results.txt")  # add-spp-block
# data5_loss = np.loadtxt("D:\HZQ\jtrafficsigns\yolov3-point/results.txt")  # 3-yolo-layers

# print(data1_loss[0])
# x = data1_loss[:,0]  # 横坐标
# y = data1_loss[:,10]  # 纵坐标
x1 = data2_loss[:,0]
y1 = data2_loss[:,10]
x2 = data3_loss[:,0]
y2 = data3_loss[:,10]
x3 = data4_loss[:,0]
y3 = data4_loss[:,10]
# x4 = data5_loss[:,0]
# y4 = data5_loss[:,10]
# x5 = data2_loss[:,0]
# y5 = data2_loss[:,10]
fig = plt.figure(figsize = (13,8))    #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
# pl.plot(x,y,'g-',label=u'yolov3-tiny')
# ‘'g‘'代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
p2 = pl.plot(x1, y1,'r-', label = u'yolov3-tiny-3l-9a-cbam-hzq-results')
p3 = pl.plot(x2, y2,'b-', label = u'yolov3-tiny-3l-9a-results')
p4 = pl.plot(x3, y3,'m-', label = u'yolov3-tiny-cbam_results')
# p5 = pl.plot(x4, y4,'c-', label = u'3-yolo-layers')

pl.legend()
#显示图例
# p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
pl.legend()
pl.xlabel(u'mAP')
pl.ylabel(u'epoch')
plt.title('Compare mAP for different models in training')
pl.show()
import os
import tkinter

import cv2
import matplotlib.pyplot as plt

'''
精度极差，没必要实现
'''



#图像颜色特征：直方图比较法法
def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.jpg':
            list_name.append(file_path)


#该函数用于统一图片大小为256*256，并且分割为16个块，返回值是16个局部图像句柄的数组
def split_Img(img, size = (64,64)):
    img = img.resize((256,256)).convert('RGB')
    w,h = img.size
    sw,sh = size
    return [img.crop((i,j,i+sw,j+sh)).copy() for i in range(0,w,sw) for j in range(0,h,sh)]

#计算两个直方图之间的相似度，h1和h2为直方图，zip表示同步遍历
def calc_Similar(h1,h2):
    return sum(1 - (0 if g==s else float(abs(g-s))/max(g,s)) for g,s in zip(h1,h2)) / len(h1)

#统计直方图，用load()载入图片的像素pix，再分别读取每个像素点的R\G\B值进行统计（分别为0-255）
#将256个颜色值的统计情况投影到0-7颜色值是一个，最后总共32个，返回R\G\B投影后的统计值数组，共32*3=96个元素
def calc_Hist(img1):
    #120张图，3.49s
    # img1 = cv2.imread('/home/hutao/桌面/image_0001.jpg', cv2.IMREAD_COLOR)
    size = img1.shape

    w=size[0]
    h=size[1]
    calcR = [0 for i in range(0,256)]
    calcG = [0 for i in range(0,256)]
    calcB = [0 for i in range(0,256)]
    for i in range(0,w):
        for j in range(0,h):
            (r,g,b) = img1[i,j]
            calcR[r] += 1
            calcG[g] += 1
            calcB[b] += 1
    calcG.extend(calcB)
    calcR.extend(calcG) #256*3

    #calc存放最终结果，32*3
    calc = [0 for i in range(0,96)]
    step = 0 #calc的下标，0~95
    start = 0 #每次统计的开始位置
    while step < 96:
        for i in range(start,start+8): #8个值为1组，统计值相加，eg：色彩值为0~7的统计值全部转换为色彩值为0的统计值
            calc[step] += calcR[i]
        start = start+8
        step += 1
    return calc


def calnum(input, data):
    ans = 0
    for i in range(0,95):
        if input[i] != 0:
            ans += abs(input[i]-data[i]/input[i])
    return ans

file = open("../Repository/Yansezhifangtu.txt")

inputpath = tkinter.filedialog.askopenfilename()
inputimg = cv2.imread(inputpath, cv2.IMREAD_COLOR)
input = calc_Hist(inputimg)
# print(input) ok
i = 0
inputlist = []
datalist = []
outputlist = []
outacc = []
flag = 0
while 1:
    line = file.readline()
    if not line:
        break
    i += 1
    if i % 2 == 1:
        name = line
    else:
        datalist = [float(x) for x in line[1:-2].split(',')]

    if len(datalist) != 96:
        continue
    num = calnum(input, datalist)
    if num < 1000:
        print(num)
    if flag == 1 and num <= 0.5:
        outputlist.append(name[0:-1])
        outacc.append(num)
    if num <= 1:
        outputlist.append(name[0:-1])
        outacc.append(num)
        flag = 1

# print(outputlist)
# print(outacc)
#
# img=inputimg
# plt.figure(num='result',figsize=(8,8))  #创建一个名为astronaut的窗口,并设置大小
#
# plt.subplot(2,2,1)     #将窗口分为两行两列四个子窗口，则可显示四幅图片
# plt.title('input img')   #第一幅图片标题
# plt.imshow(img)      #绘制第一幅图片
#
# plt.subplot(2,2,2)     #第二个子窗口
# plt.title('best: loss = '+str(outacc[0]))   #第二幅图片标题
# plt.imshow(cv2.imread("../"+outputlist[0]))
# # plt.imshow(img[:,:,0],plt.cm.gray)      #绘制第二幅图片,且为灰度图
# plt.axis('off')     #不显示坐标尺寸
#
# # print(len(outputlist))
# if len(outputlist) >= 3:
#     plt.subplot(2,2,3)     #第三个子窗口
#     plt.title('other 1: loss = '+str(outacc[1]))   #第三幅图片标题
#     plt.imshow(cv2.imread("../"+outputlist[1]))
#     plt.axis('off')     #不显示坐标尺寸
#
# if len(outputlist) >= 4:
#     plt.subplot(2,2,4)     #第四个子窗口
#     plt.title('other 2: loss = '+str(outacc[2]))   #第四幅图片标题
#     plt.imshow(cv2.imread("../"+outputlist[2]))
#     plt.axis('off')     #不显示坐标尺寸
#
# plt.show()   #显示窗口
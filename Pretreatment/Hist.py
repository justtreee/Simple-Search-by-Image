# -*- coding: UTF-8 -*-
import glob

import numpy
import random
import codecs
import os
import cv2
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


f = open("./Yansezhifangtu.txt", 'w+')
imgset = glob.glob("dataset/*/*.jpg")
for i in imgset:
    img = cv2.imread(i, cv2.IMREAD_COLOR)
    out = calc_Hist(img)
    f.write(str(i) + "\n" + str(out) + "\n")
f.close()




# path='dataset/accordion/image_0001.jpg'
# img = cv2.imread(path, cv2.IMREAD_COLOR)
#
# imgset = glob.glob("dataset/*/*.jpg")
# for i in imgset:
#     img = cv2.imread(i, cv2.IMREAD_COLOR)
#     out = calc_Hist(img)
#     print(str(out))



import tkinter

import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_moments(filename):
    img = cv2.imread(filename)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature

def cal(input,data):
    ans = 0
    for i in range(0,9):
        if input[i] == 0:
            continue
        ans += abs(input[i] - data[i])/(abs(input[i]))
    return ans
file = open("../Repository/colorjuData.txt")
# inputpath = "../dataset/ewer/image_0001.jpg" ================= 选择文件
inputpath = tkinter.filedialog.askopenfilename()
inputimg = cv2.imread(inputpath)
inputft = color_moments(inputpath)
i = 0
outputlist = []
outacc = []
flag = 0
while 1:
    line = file.readline()
    if not line:
        break
    if i % 2 == 0:
        name = line
    if i % 2 == 1:
        lstr = str(line)
        try:
            tlist = [float(x) for x in lstr.split(',')]
        except ValueError as e:
            # print("error", e, "on line", i)
            continue
        num = cal(inputft,tlist)
        # print(num)
        if flag == 1 and num < 1:
            outputlist.append(name)
            outacc.append(num)
        if num <= 0.1 and flag == 0:
            outputlist.append(name)
            outacc.append(num)
            flag = 1
    i+=1

img=inputimg
plt.figure(num='result',figsize=(8,8))  #创建一个名为astronaut的窗口,并设置大小

plt.subplot(2,2,1)     #将窗口分为两行两列四个子窗口，则可显示四幅图片
plt.title('input img')   #第一幅图片标题
plt.imshow(img)      #绘制第一幅图片

# print(outputlist[0])
plt.subplot(2,2,2)     #第二个子窗口
plt.title('best: loss = '+str(outacc[0]))   #第二幅图片标题
plt.imshow(cv2.imread("../"+outputlist[0][0:-1]))
# plt.imshow(img[:,:,0],plt.cm.gray)      #绘制第二幅图片,且为灰度图
plt.axis('off')     #不显示坐标尺寸

# print(len(outputlist))
if len(outputlist) >= 3:
    plt.subplot(2,2,3)     #第三个子窗口
    plt.title('other 1: loss = '+str(outacc[1]))   #第三幅图片标题

    # print(outputlist[1][0:-1])
    plt.imshow(cv2.imread("../"+outputlist[1][0:-1]))
    plt.axis('off')     #不显示坐标尺寸

if len(outputlist) >= 4:
    plt.subplot(2,2,4)     #第四个子窗口
    plt.title('other 2: loss = '+str(outacc[2]))   #第四幅图片标题
    plt.imshow(cv2.imread("../"+outputlist[2][0:-1]))
    plt.axis('off')     #不显示坐标尺寸

plt.show()   #显示窗口
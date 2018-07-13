import tkinter

import cv2
import matplotlib.pyplot as plt

def ft(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(img_gray)
    humoments = cv2.HuMoments(moments)
    return humoments

def cal(input, data):
    ans = 0
    for i in range(0,6):
        if input[i] == 0:
            continue
        ans += abs(input[i] - data[i])/abs(input[i])
    return ans

file = open("../Repository/ShapeNchangeData.txt")
inputpath = tkinter.filedialog.askopenfilename()
inputimg = cv2.imread(inputpath)

inlist = ft(inputimg)

i = 0
linelist = []
outputlist = []
outacc = []
flag = 0
loop = 0
while 1:
    line = file.readline()
    if not line:
        break
    i += 1
    case = i % 8
    if case == 1:
        name = line[1:-2]
    elif case != 1 and case != 0:
        linelist.append(float(line[2:-2]))
    elif case == 0:
        linelist.append(float(line[2:-3]))

    # print(name)
    # print(linelist)
    if case == 0 and len(linelist) == 7:
        num = cal(inlist, linelist)
        if flag == 1 and num < 5:
            outputlist.append(name)
            outacc.append(num)
        if num < 0.1 and flag == 0:
            outputlist.append(name)
            outacc.append(num)
            flag = 1
    if case == 0 and len(linelist) == 7:
        linelist = []


# print(outputlist)
img=inputimg
plt.figure(num='result',figsize=(8,8))  #创建一个名为astronaut的窗口,并设置大小

plt.subplot(2,2,1)     #将窗口分为两行两列四个子窗口，则可显示四幅图片
plt.title('input img')   #第一幅图片标题
plt.imshow(img)      #绘制第一幅图片

# print(outputlist[0])
plt.subplot(2,2,2)     #第二个子窗口
plt.title('best: loss = '+str(outacc[0]))   #第二幅图片标题
plt.imshow(cv2.imread("../"+outputlist[0]))
# plt.imshow(img[:,:,0],plt.cm.gray)      #绘制第二幅图片,且为灰度图
plt.axis('off')     #不显示坐标尺寸

# print(len(outputlist))
if len(outputlist) >= 3:
    plt.subplot(2,2,3)     #第三个子窗口
    plt.title('other 1: loss = '+str(outacc[1]))   #第三幅图片标题

    # print(outputlist[1][0:-1])
    plt.imshow(cv2.imread("../"+outputlist[1]))
    plt.axis('off')     #不显示坐标尺寸

if len(outputlist) >= 4:
    plt.subplot(2,2,4)     #第四个子窗口
    plt.title('other 2: loss = '+str(outacc[2]))   #第四幅图片标题
    plt.imshow(cv2.imread("../"+outputlist[2]))
    plt.axis('off')     #不显示坐标尺寸

plt.show()   #显示窗口
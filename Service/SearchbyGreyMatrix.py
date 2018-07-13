import tkinter

import cv2
import math
import matplotlib.pyplot as plt

file = open("../Repository/GreyMatrixData.txt")
inputpath = tkinter.filedialog.askopenfilename()
inputimg = cv2.imread(inputpath)

#定义最大灰度级数
gray_level = 16

def maxGrayLevel(img):
    max_gray_level=0
    (height,width)=img.shape
    # print height,width
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level+1

def getGlcm(input,d_x,d_y):
    srcdata=input.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height,width) = input.shape

    max_gray_level=maxGrayLevel(input)

    #若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i]*gray_level / max_gray_level

    for j in range(height-d_y):
        for i in range(width-d_x):
             rows = srcdata[j][i]
             cols = srcdata[j + d_y][i+d_x]
             ret[rows][cols]+=1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j]/=float(height*width)

    return ret

def feature_computer(p):
    Con=0.0
    Eng=0.0
    Asm=0.0
    Idm=0.0
    for i in range(gray_level):
        for j in range(gray_level):
            Con+=(i-j)*(i-j)*p[i][j]
            Asm+=p[i][j]*p[i][j]
            Idm+=p[i][j]/(1+(i-j)*(i-j))
            if p[i][j]>0.0:
                Eng+=p[i][j]*math.log(p[i][j])
    return Asm,Con,-Eng,Idm

def test(img):

    try:
        img_shape=img.shape
    except:
        print ('imread error')
        return -1

    img=cv2.resize(img,(int(img_shape[1]/2),int(img_shape[0]/2)),interpolation=cv2.INTER_CUBIC)

    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0=getGlcm(img_gray, 1,0)
    #glcm_1=getGlcm(src_gray, 0,1)
    #glcm_2=getGlcm(src_gray, 1,1)
    #glcm_3=getGlcm(src_gray, -1,1)

    asm,con,eng,idm=feature_computer(glcm_0)

    return asm,con,eng,idm

def calc(input, data):
    res = 0
    for i in range (0,4):
        if input[i] == 0:
            continue
        res += abs(float(input[i]-data[i]))/abs(float(input[i]))
    return res
inputlist = (test(inputimg))

i = 1
outputlist = []
outacc = []
flag = 0
while 1:
    line = file.readline()
    if i % 2 == 1:
        name = line
    if i % 2 == 0:
        if not line:
            break
        lstr = str(line)
        length = len(lstr)
        lstr = lstr[1:length-2]
        tlist = [float(x) for x in lstr.split(',')]
        num = calc(inputlist, tlist)

        if flag == 1 and num <= 0.5:
            outputlist.append(name[1:-2])
            outacc.append(num)
        if num <= 0.001 and flag == 0:
            # print([num, name])
            outputlist.append(name[1:-2])
            outacc.append(num)
            flag = 1

    i+=1

# print(outputlist)

img=inputimg
plt.figure(num='result',figsize=(8,8))  #创建一个名为astronaut的窗口,并设置大小

plt.subplot(2,2,1)     #将窗口分为两行两列四个子窗口，则可显示四幅图片
plt.title('input img')   #第一幅图片标题
plt.imshow(img)      #绘制第一幅图片

plt.subplot(2,2,2)     #第二个子窗口
plt.title('best: loss = '+str(outacc[0]))   #第二幅图片标题
plt.imshow(cv2.imread("../"+outputlist[0]))
# plt.imshow(img[:,:,0],plt.cm.gray)      #绘制第二幅图片,且为灰度图
plt.axis('off')     #不显示坐标尺寸

# print(len(outputlist))
if len(outputlist) >= 3:
    plt.subplot(2,2,3)     #第三个子窗口
    plt.title('other 1: loss = '+str(outacc[1]))   #第三幅图片标题
    plt.imshow(cv2.imread("../"+outputlist[1]))
    plt.axis('off')     #不显示坐标尺寸

if len(outputlist) >= 4:
    plt.subplot(2,2,4)     #第四个子窗口
    plt.title('other 2: loss = '+str(outacc[2]))   #第四幅图片标题
    plt.imshow(cv2.imread("../"+outputlist[2]))
    plt.axis('off')     #不显示坐标尺寸

plt.show()   #显示窗口
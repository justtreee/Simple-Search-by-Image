# 一． 系统设计
基于内容的图像检索系统（Content Based Image Retrieval, 以下简称 CBIR），是计算机视 觉领域中关注大规模数字图像内容检索的研究分支。典型的 CBIR 系统，允许用户输入一张图像，在图像数据库（或本地机、或网络）中查找具有相同或相似内容的其它图片。本实训
的基本功能要求是实现基于视觉特征的图像检索。具体包括：
1. 实现基于颜色信息的图像 检索，可通过颜色直方图、颜色矩、颜色一致性矢量等方法来实现。
2. 实现基于纹理特征 的图像检索，可从四个方面进行：统计法、结构法、模型法、频谱法。
3. 实现基于形状特 征的图像检索，可分别从图像的边缘信息和区域信息来实现。
4. 实现基于综合信息的图像检索。

用户通过GUI，选择使用哪一种方法来图像检索，并通过选择文件的方式打开图片，最后在结果界面上显示输入图片，最佳图片，和两张较相似的图片。

# 二． 所采用算法思想
## 1. 基于颜色特征检索
### HSV 中心距法
> colorjuData.txt 保存的是 HSV 中心距法的特征值

颜色矩（color moments）是由Stricker 和Orengo所提出的一种非常简单而有效的颜色特征。这种方法的数学基础在于图像中任何的颜色分布均可以用它的矩来表示。此外，由于颜色分布信息主要集中在低阶矩中，因此仅采用颜色的一阶矩（mean）、二阶矩（variance）和三阶矩（skewness）就足以表达图像的颜色分布。与颜色直方图相比，该方法的另一个好处在于无需对特征进行向量化。因此，图像的颜色矩一共只需要9个分量（3个颜色分量，每个分量上3个低阶矩），与其他的颜色特征相比是非常简洁的。在实际应用中，为避免低次矩较弱的分辨能力，颜色矩常和其它特征结合使用，而且一般在使用其它特征前，起到过滤缩小范围（narrow down）的作用。HSV 中心距法是基于HSV空间的因此需要将RGB空间转换为HSV空间[1]。

### 直方图相交法
> Yansezhifangtu.txt 保存的是直方图相交法的特征值。

利用图像的特征描述图像，可借助特征的统计直方图。图像特征的统计直方图实际是一个1-D的离散函数[2],
上式中k代表图像的特征取值，L是特征可取值个数，是图像中具有特征值为k的像素的个数，N是图像像素的总数，一个示例如下图：其中有8个直方条，对应图像中的8种灰度像素在总像素中的比例。

得到图像特征的统计直方图后，不同图像之间的特征匹配可借助计算直方图间的相似度量来进行，以下介绍几种常见的直方图的相似度量方法：
1. 直方图相交法
另这里写图片描述分别为两幅图像某一特征的统计直方图，则两图像之间的匹配值P(Q, D)可借助直方图相交来实现，

2. 直方图匹配法
直方图间的距离可使用一般的欧式距离函数这里写图片描述来衡量
## 2. 基于纹理特征检索
### 灰度矩阵法
> greymatrixData.txt 保存的是灰度矩阵法的特征值。

灰度共生矩阵法(GLCM, Gray-level co-occurrence matrix)，就是通过计算灰度图像得到它的共生矩阵，然后透过计算该共生矩阵得到矩阵的部分特征值，来分别代表图像的某些纹理特征（纹理的定义仍是难点）。灰度共生矩阵能反映图像灰度关于方向、相邻间隔、变化幅度等综合信息，它是分析图像的局部模式和它们排列规则的基础。
对于灰度共生矩阵的理解，需要明确几个概念：方向，偏移量和灰度共生矩阵的阶数[3]。

- 方向：一般计算过程会分别选在几个不同的方向来进行，常规的是水平方向0°，垂直90°，以及45°和135°；
- 步距d：中心像元（在下面的例程中进行说明）；
- 灰度共生矩阵的阶数：与灰度图像灰度值的阶数相同，即当灰度图像灰度值阶数为N时，灰度共生矩阵为N × N的矩阵；

灰度共生矩阵（Gray-Level Co-occurrence Matrix，GLCM）统计了灰度图中像素间的灰度值分布规律以区分不同的纹理。

灰度共生矩阵可以定义为一个灰度为[Math Processing Error]的像素点与另一个与之对应位置上的像素点的灰度值为[Math Processing Error]的概率。那么所有估计的值可以表示成一个矩阵的形式，以此被称为灰度共生矩阵。如：根据图像中任意一点 [Math Processing Error] 的灰度值和它所对应的点 [Math Processing Error] 的灰度值可以得到一个灰度值组合 [Math Processing Error]。统计整福图像每一种灰度值组合出现的概率矩阵 [Math Processing Error] 即为灰度共生矩阵。

由于灰度共生矩阵的维度较大，一般不直接作为区分纹理的特征，而是基于它构建的一些统计量作为纹理分类特征。例如[Math Processing Error]曾提出了14种基于灰度共生矩阵计算出来的统计量：能量、熵、对比度、均匀性、相关性、方差、和平均、和方差、和熵、差方差、差平均、差熵、相关信息测度以及最大相关系数[5]。
## 3. 基于形状特征检索
### 形状的不变矩法
> ShapeNchangeData.txt 保存的是基于形状的不变矩法特征值。

几何矩是由Hu(Visual pattern recognition by moment invariants)在1962年提出的，具有平移、旋转和尺度不变性。

由Hu矩组成的特征量对图片进行识别，优点就是速度很快，缺点是识别率比较低，我做过手势识别，对于已经分割好的手势轮廓图，识别率也就30%左右，对于纹理比较丰富的图片，识别率更是不堪入眼，只有10%左右。这一部分原因是由于Hu不变矩只用到低阶矩（最多也就用到三阶矩），对于图像的细节未能很好的描述出来，导致对图像的描述不够完整。

Hu不变矩一般用来识别图像中大的物体，对于物体的形状描述得比较好，图像的纹理特征不能太复杂，像识别水果的形状，或者对于车牌中的简单字符的识别效果会相对好一些[4]。
### 形状边缘直方图法
> shapeHistogramData.txt 保存的是基于形状边缘直方图法特征值。

边缘分布直方图作为一种简单有效的形状表示方法，在目标检测，目标识别等方面一直有着广泛的应用。

边缘检测根本上说就是通过比较图像各点的边缘方向信息来进行识别。

一般我们采用MPEG-7对图像边缘的定义对图像中各点的边缘信息进行分类。

MPEG-7将图像边缘分为五个方向，水平、竖直、斜45度、斜135度和无方向。这五种方向分别对应五个不同的边缘描述子。

这里我们介绍一个分块的边缘直方图匹配方法，该方法适合于在大规模的图象数据中找到需要的结果，搜索速度较快。

首先对图像进行分块，然后使用图1中的五种边缘描述子对各块的边缘方向信息进行提取，并画出直方图，在数据库中搜索与各小块中直方图匹配的数据。该方法运算速度快，适合大规模图像的查找，但是没能考虑图块的位置问题，不适用与精确的图像搜索。

# 三． 详细实现过程
## HSV 中心距法
上文介绍过HSV中心距法的算法，首先就是将RGB空间转换为HSV空间，得到三个通道的值之后进行运算。每张图片获得九维的数据。
```python
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
```

输出，得到形如下文的数据检索库：
```
dataset/brain/image_0022.jpg
36.1909235209,42.2138095238,210.893708514,53.3302508431,62.1570734081,60.4449336241,64.8843752318,79.7674844775,71.5411568249
```

## 直方图相交法
图像颜色特征：直方图比较法

1. 读入文件
```python
def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.jpg':
            list_name.append(file_path)
```

2. 该函数用于统一图片大小为256*256，并且分割为16个块，返回值是16个局部图像句柄的数组
```python
def split_Img(img, size = (64,64)):
    img = img.resize((256,256)).convert('RGB')
    w,h = img.size
    sw,sh = size
    return [img.crop((i,j,i+sw,j+sh)).copy() for i in range(0,w,sw) for j in range(0,h,sh)]
```
3. 计算两个直方图之间的相似度，h1和h2为直方图，zip表示同步遍历
```python
def calc_Similar(h1,h2):
    return sum(1 - (0 if g==s else float(abs(g-s))/max(g,s)) for g,s in zip(h1,h2)) / len(h1)
```
4. 统计直方图，用load()载入图片的像素pix，再分别读取每个像素点的R\G\B值进行统计（分别为0-255）将256个颜色值的统计情况投影到0-7颜色值是一个，最后总共32个，返回R\G\B投影后的统计值数组，共32*3=96个元素
```python
def calc_Hist(img1):
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
```

## 灰度矩阵法
-  角二阶矩（Angular Second Moment, ASM)
角二阶矩又称能量，是图像灰度分布均匀程度和纹理粗细的一个度量，反映了图像灰度分布均匀程度和纹理粗细度。当图像纹理均一规则时，能量值较大；反之灰度共生矩阵的元素值相近，能量值较小。
- 熵（Entropy, ENT)
熵度量了图像包含信息量的随机性，表现了图像的复杂程度。当共生矩阵中所有值均相等或者像素值表现出最大的随机性时，熵最大。
- 对比度
对比度反应了图像的清晰度和纹理的沟纹深浅。纹理越清晰反差越大对比度也就越大。
- 反差分矩阵（Inverse Differential Moment, IDM)
反差分矩阵又称逆方差，反映了纹理的清晰程度和规则程度，纹理清晰、规律性较强、易于描述的，值较大。


1. 通过最大灰度级数，将原图转化为灰度图
```python
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
```
2. 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
```python
def getGlcm(input,d_x,d_y):
    srcdata=input.copy()
    ret=[[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height,width) = input.shape

    max_gray_level=maxGrayLevel(input)

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
```
3. 计算的到特征值
```python
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
```
4. 对输入图像进行预处理之后，调用上文函数进行处理得到特征值，每张图片有四维的特征。
```python
def test(img, i):
    try:
        img_shape=img.shape
    except:
        print ('imread error')
        return -1
    img=cv2.resize(img,(int(img_shape[1]/2),int(img_shape[0]/2)),interpolation=cv2.INTER_CUBIC)
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm_0=getGlcm(img_gray, 1,0)
    asm,con,eng,idm=feature_computer(glcm_0)
    f.write("{" +i+"}\n["+str(asm)+","+str(con)+","+str(eng)+","+str(idm)+"]\n")
```
输出形如下文：
```
{dataset\accordion\image_0001.jpg}
[0.10931862721893505,13.442923076923089,3.91634820329216,0.5884026464174514]
```
## 形状的不变矩法
主要使用了OpenCV自带的函数：
```python
moments = cv2.moments(img)
humoments = cv2.HuMoments(moments)
return humoments
```

## 形状边缘直方图法
1. 生成二维高斯分布矩阵，并转化为灰度图
```python
def ft(img):
    sigma1 = sigma2 = 1
    sum = 0

    gaussian = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - 3) / np.square(sigma1)
                                                + (np.square(j - 3) / np.square(sigma2)))) / (2 * math.pi * sigma1 * sigma2)
            sum = sum + gaussian[i, j]

    gaussian = gaussian / sum
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

```
2. 使用高斯滤波
```python
    gray = rgb2gray(img)
    W, H = gray.shape
    new_gray = np.zeros([W - 5, H - 5])
    for i in range(W - 5):
        for j in range(H - 5):
            new_gray[i, j] = np.sum(gray[i:i + 5, j:j + 5] * gaussian)  # 与高斯矩阵卷积实现滤波

    # plt.imshow(new_gray, cmap="gray")
```
3. 增强 通过求梯度幅值
```python
    W1, H1 = new_gray.shape
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1 - 1, H1 - 1])
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # 图像梯度幅值作为图像强度值

    # plt.imshow(d, cmap="gray")
```
4. 非极大值抑制 NMS
```python
    W2, H2 = d.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[W2 - 1, :] = NMS[:, 0] = NMS[:, H2 - 1] = 0
    for i in range(1, W2 - 1):
        for j in range(1, H2 - 1):
            if d[i, j] == 0:
                NMS[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = d[i, j]
                # 如果Y方向幅度值较大
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]
                # 如果X方向幅度值较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    # 如果x,y方向梯度符号相同
                    if gradX * gradY > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    # 如果x,y方向梯度符号相反
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp
                else:
                    NMS[i, j] = 0
```
5. 双阈值算法检测、连接边缘
```python
    W3, H3 = NMS.shape
    DT = np.zeros([W3, H3])
    # 定义高低阈值
    TL = 0.2 * np.max(NMS)
    TH = 0.3 * np.max(NMS)
    for i in range(1, W3 - 1):
        for j in range(1, H3 - 1):
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
                or (NMS[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1
    size = DT.shape
```
6. 将整张图片分为16个分块，计算每一个分块的黑白像素个数。
```python
    w=size[0]
    h=size[1]
    wnew=int(w/4)
    hnew=int(h/4)
    #calc存放最终结果，256
    step = 0 #calcnew的下标，0~63
    start = 0 #每次统计的开始位置
    calc = [0 for i in range(0,16)]
    count=0
    i=0
    j=0
    k=0
    for i in range(0,w):
        for j in range(0,h):
            if i>=0 and i<=wnew and j>=0 and j<=hnew and DT[i][j]==1:
                calc[0]=calc[0]+1
            if i>=wnew and i<=2*wnew and j>=0 and j<=hnew and DT[i][j]==1:
                calc[1]=calc[1]+1
            # .....篇幅有限省略部分代码.....
            if i>=3*wnew and i<=4*wnew and j>=3*hnew and j<=4*hnew and DT[i][j]==1:
                calc[15]=calc[15]+1
    return calc
```

# 四． 实验结果分析
本部分的实验都围绕以下三张图片展开
![default](https://user-images.githubusercontent.com/15559340/42620530-c032224c-85ed-11e8-9ab4-f8b970fa7b84.PNG)
## 1. HSV 中心距法
<div align=center>

![brain](https://user-images.githubusercontent.com/15559340/42619851-c8ae9240-85eb-11e8-964b-c1d114dc796a.png)
图1.1 Brain034-good_example
![badexample](https://user-images.githubusercontent.com/15559340/42619849-c832d6b4-85eb-11e8-854b-fb14a5e14985.png)
图1.2 Brain007-bad_example
这是一个反例。可以看到输入的图片不同于图片库中的其他大脑图片，呈现多彩的感觉，并且以蓝紫色偏多，这在HSV colorJu 方法下是不准确的。检索之后发现返回的other图片不是大脑，但很明显图片风格是相近的。
![normal](https://user-images.githubusercontent.com/15559340/42619854-c9665d1c-85eb-11e8-8b4f-c520e8e43593.png)
图1.3 Airplanes001
这个可以作为一个常规的例子，既有合理的相似图片返回，也有完全不一样的图片。但只看颜色的话，多张图片的色彩相近。这也表现了该方法的缺陷。
</div>

## 2. 灰度矩阵法

<div align=center>

![1](https://user-images.githubusercontent.com/15559340/42620360-5659e76a-85ed-11e8-93d7-03b02d2d4860.png)
图2.1 Brain007
可以看到下方两张图片的纹理与原图接近，确实体现了灰度矩阵法的特性。
![2](https://user-images.githubusercontent.com/15559340/42620362-56ca8fb0-85ed-11e8-8403-558d94ff5bc0.png)
图2.2 Brain034
![3](https://user-images.githubusercontent.com/15559340/42620363-5742438e-85ed-11e8-846b-ee69c8fcc9b7.png)
图2.3 Airplanes001
与第一种方法HSV中心矩法相比，检索出来的图片更接近。

</div>

## 3. 形状的不变矩法
<div align=center>

![2](https://user-images.githubusercontent.com/15559340/42620421-7e6037a0-85ed-11e8-897e-15ebcdf00b3e.png)
图3.1
除了最佳匹配，其他的都不准确，但也能看出来他们的形状是相似的
![3](https://user-images.githubusercontent.com/15559340/42620423-7edb5340-85ed-11e8-8d06-98ad8d5eeeda.png)
图3.2
除了最佳匹配，其他图片的loss值都较高
![1](https://user-images.githubusercontent.com/15559340/42620420-7de77cf2-85ed-11e8-84fb-577f0aa38826.png)
图3.3
也许是因为原图的尾翼、机头轮廓明显，检索出来的图片的尾翼机头与原图非常相似。
</div>

## 4. 形状边缘直方图法
<div align=center>

![edgeimage1](https://user-images.githubusercontent.com/15559340/42620461-93ff5294-85ed-11e8-821f-8b392658cca2.png)
![edgeimage3](https://user-images.githubusercontent.com/15559340/42620464-9451c984-85ed-11e8-971d-1c6651ef50d2.png)
![egdeimage2](https://user-images.githubusercontent.com/15559340/42620465-949b8c5e-85ed-11e8-9624-a5042f1c4ec5.png)
图4.1-2
这里显示一下中间变量，使用canny算子，可以看到边缘提取很明显。

![hist](https://user-images.githubusercontent.com/15559340/42620468-95d715de-85ed-11e8-8c48-c09aed2a056b.PNG)

![shapehistogram1](https://user-images.githubusercontent.com/15559340/42620469-9650bf24-85ed-11e8-9d5a-65ef57f3bebc.PNG)
![shapehistogram2](https://user-images.githubusercontent.com/15559340/42620470-96cad78c-85ed-11e8-99a0-8f14efe6d653.PNG)
图4.3-4
轮廓图相应的直方图
![2](https://user-images.githubusercontent.com/15559340/42620455-9351a676-85ed-11e8-9dea-589ee277dc14.png)
![3](https://user-images.githubusercontent.com/15559340/42620457-939cef82-85ed-11e8-9086-65e81570670b.png)
![1](https://user-images.githubusercontent.com/15559340/42620454-92afaaa6-85ed-11e8-9940-b0ab5954adf5.png)
图4.5-7 结果
</div>


# 五． 实训总结和心得
本次实验是我第一次接触python环境下的图像处理。通过几天的实训，实现了基于内容的图像检索技术是对输入的图像进行分析并分类统一建模，提取其颜色、形状、
纹理、轮廓和空间位置等特征，建立特征索引, 存储于特征数据库中。检索时，用户提交查
询的源图像，通过用户接口设置查询条件，可以采用一种或几种的特征组合来表示，然后在
图像数据库中提取出查询到的所需关联图像，按照相似度从大到小的顺序，反馈给用户。

根据已有数据集，首先进行预处理缓存到硬盘中，再分析并得出合理的检索方法，实现了四种不同的特征值提取方法。提高了个人编程能力与解决问题能力。

# 六． 参考文献
[1] 颜色矩原理及Python实现 https://www.cnblogs.com/klchang/p/6512310.html

[2] 图像相似度检测之直方图相交 http://www.voidcn.com/article/p-crcxchuk-px.html

[3] 纹理特征提取方法：LBP, 灰度共生矩阵 https://blog.csdn.net/ajianyingxiaoqinghan/article/details/71552744

[4] Hu不变矩原理及opencv实现 https://blog.csdn.net/qq_26898461/article/details/47123405

[5] 灰度共生矩阵（GLCM） https://blog.csdn.net/kmsj0x00/article/details/79463376

[6] Image moment https://en.wikipedia.org/wiki/Image_moment#Rotation_invariant_moments
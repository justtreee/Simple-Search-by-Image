import sys
import os
import tkinter
top=tkinter.Tk()
def colorju():
    os.system('python SearchbyColorju.py')

def greyMatrix():
    os.system('python SearchbyGreyMatrix.py')

def shapeHist():
    os.system('python SearchbyShapeHist.py')

def shapeNchange():
    os.system('python SearchbyShapeNchange.py')

top.geometry('300x230+500+200')

B1=tkinter.Button(top,text="基于颜色特征检索：HSV 中心距法",command= colorju)
B2=tkinter.Button(top,text="基于纹理特征检索：灰度矩阵法",command= greyMatrix)
B3=tkinter.Button(top,text="基于形状特征检索：形状边缘直方图法",command= shapeHist)
B4=tkinter.Button(top,text="基于形状特征检索：形状的不变矩法",command= shapeNchange)
B5=tkinter.Button(top,text="---混合检索---",command= greyMatrix)

B1.pack(fill=tkinter.X, padx=5,pady=6)
B2.pack(fill=tkinter.X, padx=5,pady=6)
B3.pack(fill=tkinter.X, padx=5,pady=6)
B4.pack(fill=tkinter.X, padx=5,pady=6)
B5.pack(fill=tkinter.X, padx=5,pady=6)
top.mainloop()
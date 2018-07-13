#!/usr/bin/python
# -*- coding: UTF-8 -*-

# hu不变矩
# https://blog.csdn.net/qq_23926575/article/details/80624630
import glob
import cv2

def test(img):
    moments = cv2.moments(img)
    humoments = cv2.HuMoments(moments)
    return humoments

if __name__=='__main__':
    f = open("./ShapeNchangeData.txt", 'w+')
    imgset = glob.glob("dataset/*/*.jpg")
    for i in imgset:
        img = cv2.imread(i)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = "{" + str(i) + "}\n" + str(test(img_gray)) + "\n"
        print(out)
        # f.write(out)
    f.close()
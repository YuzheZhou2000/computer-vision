# -*- coding: utf-8 -*-
from os import pathsep
from PIL import Image, ImageDraw, ImageFont
import traceback
import numpy
import cv2
import math
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.core.arrayprint import printoptions


def Polar2Cartesian(theata):

    pi = 3.14
    radian = pi * theata/180
    return math.tan(radian)


def addTextWaterMark(text):
    # 加载需要添加水印的图片
    img = cv2.imread('image/demo.jpg')

    # 添加文字水印 参数：原始图像-添加水印的字符串-坐标=字体-字号-颜色-粗细
    img2 = cv2.putText(img, text, (100, 100),
                       cv2.LINE_AA, 2, (0, 249, 249), 4)

    # 保存图片
    cv2.imwrite('output/waterMark.jpg', img2)

    # 在窗口显示
    cv2.imshow('img', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def addChaneseWaterMark(text):
    # 读取图片信息
    img = cv2.imread('image/demo.jpg')
    # 判断是否是openCV图片类型
    if (isinstance(img, numpy.ndarray)):
        # 转化成PIL类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # 字体的格式
    textSize = 80
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    left = 100
    top = 100
    textColor = (168, 121, 103)
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV类型
    img2 = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    # 保存图片
    cv2.imwrite('output/wj.jpg', img2)
    cv2.imshow('img', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def addPictureWAterMark():
    try:
        # 打开图片
        img = Image.open('image/demo.jpg')
        # 加载水印图片
        waterMark = Image.open('image/iu-lizhien.jpeg')
        # 复制一个尺寸大小一样的图层
        img_size = img.size
        wtm_size = waterMark.size
        # 保证水印图片的大小小于目标图片大小
        waterMark.resize(
            tuple(map(lambda x: int(x * 0.005), waterMark.size)))
        # 设定水印的位置
        wtm_position = (img_size[0] - wtm_size[0], img_size[1]-img_size[1])
        # 新建一个图层
        layer = Image.new('RGBA', img.size)
        # 将水印添加到图层
        layer.paste(waterMark, wtm_position)
        mark_img = Image.composite(layer, img, layer)
        mark_img.save('output/newmark.jpg')
        print('done')
    except Exception as e:
        print(traceback.print_exc())


def drawLine2PictureDemo(imgName):
    # 加载图片 ori
    ori = cv2.imread(imgName)
    theta = 40
    # 获得原始图像的宽和高
    width, height = ori.shape[0], ori.shape[1]
    img = numpy.zeros((height, width))
    # 直角坐标系下的斜率
    slope = Polar2Cartesian(theta)
    print("slope: ", slope)
    if theta < 90:
        min_offset = int(0 - slope * width)
        max_offset = height - 1
    else:
        min_offset = 1
        max_offset = 2 * height - int(slope * width)
    # 绘图
    for offset in range(min_offset, max_offset, 30):
        if theta < 45:
            for i in range(0, width):
                y = int(slope * i) + offset
                if y >= height:
                    break
                # idx += 1
                img[y, i] = 255
        elif theta <= 90:
            """
            x = 1/slop
            """
            for i in range(0, height):
                x = int((i - offset) / slope)
                if x >= width or x < 0:
                    continue
                # idx += 1
                img[i, x] = 255
        elif theta <= 135:
            for i in range(0, height):
                x = int((i - offset) / slope)
                if x >= width or x < 0:
                    continue
                # idx += 1
                img[i, x] = 255
        else:
            for i in range(0, width):
                y = int(slope * i) + offset
                if (y >= height or y < 0) and offset < height:
                    break
                if (y >= height or y < 0) and offset >= height:
                    continue
                # idx += 1
                img[y, i] = 255
     # 保存图片
    cv2.imwrite('output/writeLine.jpg', img)
    return img


def drawUseOpencv():
    img = cv2.imread('image/demo.jpg')
    # shape返回的是一个tuple元组，第一个元素表示图像的高度，第二个表示图像的宽度，第三个表示像素的通道数。
    size = img.shape  # 700*700*3

    # pt1表示起始位置  pt2 表示结束位置 分别画 直线、矩形、圆、椭圆
    cv2.line(img, pt1=(50, 30), pt2=(400, 30),
             color=(30, 30, 30), thickness=0.01)

    # cv2.rectangle(img, pt1=(300, 0), pt2=(500, 150),
    #               color=(0, 255, 0), thickness=0.1)
    # cv2.circle(img, center=(400, 60), radius=60,
    #            color=(0, 255, 255), thickness=1)
    # cv2.ellipse(img, center=(256, 256), axes=(100, 50), angle=0,
    #             startAngle=0, endAngle=180, color=(0, 255, 255), thickness=1)

    cv2.imshow('aaa', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def drawUseMatplotlib():
    # 文件地址
    img_path = 'image/demo.jpg'
    # 打开图片
    img = Image.open(img_path)  # 700*700
    # print(type(img), img.size)

    im = np.array(img, dtype=np.uint8)
    print(type(im), im.shape)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    """
        Parameters
        ----------
        xy : (float, float)
            The anchor point.
        width : float
            Rectangle width.
        height : float
            Rectangle height.
        angle : float, default: 0
            Rotation in degrees anti-clockwise about *xy*.

        Other Parameters
        ----------------
        **kwargs : `.Patch` properties
            %(Patch)s
    """
    rect = patches.Rectangle(
        (100, 600), 700, 0, linewidth=2, edgecolor=[0, 0, 0, 0.7], facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

    rect = patches.Rectangle(
        (60, 200), 40, 0, linewidth=2, edgecolor=[0, 0, 0, 0.7], facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    """
        Parameters
        ----------
        xy : (float, float)
            xy coordinates of ellipse centre.
        width : float
            Total length (diameter) of horizontal axis.
        height : float
            Total length (diameter) of vertical axis.
        angle : float, default: 0
            Rotation in degrees anti-clockwise.

        Notes
        -----
        Valid keyword arguments are:

        %(Patch)s
    """
    rect = patches.Ellipse((300, 300), 20, 50, 90, linewidth=2, edgecolor=[
                           0, 0, 0, 0.3], facecolor='none')
    ax.add_patch(rect)

    plt.savefig('output/addline.jpg', dpi=1000, bbox_inches="tight")
    plt.show()


def drawWaterMarkGreyOfEllipse(path):
    '''
    function:
        use  matplotlib.patches to make watermark in grey 

    Parameters: 
        path: path of picture

    return:
        nil
    '''

    # Load picture
    img = Image.open(path)  # 1300*1800
    im = np.array(img, dtype=np.uint8)  # (1800, 1300, 4)
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Circle patch
    """
        Parameters
        ----------
        xy : (float, float)
            xy coordinates of ellipse centre.
        width : float
            Total length (diameter) of horizontal axis.
        height : float
            Total length (diameter) of vertical axis.
        angle : float, default: 0
            Rotation in degrees anti-clockwise.

        Notes
        -----
        Valid keyword arguments are:

        %(Patch)s
    """

    # 250个样本圈    linewidth =  0.03
    for transparent in range(0, 10):
        # 控制透明度 十个等级
        for grey in range(0, 25):
            # 控制灰度 25个等级
            rect = patches.Ellipse((20+transparent*30, 36+grey*72), 26, 70, 0, linewidth=0.03, edgecolor=[
                grey/25, grey/25, grey/25, transparent*1.0/10], facecolor='none')
            ax.add_patch(rect)

    # 分界线
     # Create a Rectangle patch
    """
        Parameters
        ----------
        xy : (float, float)
            The anchor point.
        width : float
            Rectangle width.
        height : float
            Rectangle height.
        angle : float, default: 0
            Rotation in degrees anti-clockwise about *xy*.

        Other Parameters
        ----------------
        **kwargs : `.Patch` properties
            %(Patch)s
    """
    rect = patches.Rectangle(
        (320, 0), 0, 1800, linewidth=0.3, edgecolor=[0, 0, 0, 1], facecolor='none')
    ax.add_patch(rect)

    # 250个样本圈    linewidth =  0.04
    for transparent in range(0, 10):
        # 控制透明度 十个等级
        for grey in range(0, 25):
            # 控制灰度 25个等级
            rect = patches.Ellipse((320+transparent*30, 36+grey*72), 26, 70, 0, linewidth=0.04, edgecolor=[
                grey/25, grey/25, grey/25, transparent*1.0/10], facecolor='none')
            ax.add_patch(rect)

    # 分界线
    rect = patches.Rectangle(
        (620, 0), 0, 1800, linewidth=0.3, edgecolor=[0, 0, 0, 1], facecolor='none')
    ax.add_patch(rect)

    # 250个样本圈    linewidth =  0.05
    for transparent in range(0, 10):
        # 控制透明度 十个等级
        for grey in range(0, 25):
            # 控制灰度 25个等级
            rect = patches.Ellipse((620+transparent*30, 36+grey*72), 26, 70, 0, linewidth=0.05, edgecolor=[
                grey/25, grey/25, grey/25, transparent*1.0/10], facecolor='none')
            ax.add_patch(rect)

    # 分界线
    rect = patches.Rectangle(
        (920, 0), 0, 1800, linewidth=0.3, edgecolor=[0, 0, 0, 1], facecolor='none')
    ax.add_patch(rect)

    # 250个样本圈    linewidth =  0.06
    for transparent in range(0, 10):
        # 控制透明度 十个等级
        for grey in range(0, 25):
            # 控制灰度 25个等级
            rect = patches.Ellipse((920+transparent*30, 36+grey*72), 26, 70, 0, linewidth=0.06, edgecolor=[
                grey/25, grey/25, grey/25, transparent*1.0/10], facecolor='none')
            ax.add_patch(rect)

    # 保存图片
    # 取消坐标轴显示
    plt.axis('off')
    plt.savefig('output/addEllipse.jpg', dpi=2000, bbox_inches="tight")
    # plt.show()
    print('drawWaterMarkGreyOfEllipse done')
    return


def drawWaterMarkColorOfEllipse(path):
    '''
    function:
        use  matplotlib.patches to make watermark in color

    Parameters: 
        path: path of picture

    return:
        nil
    '''
    a = 0.1
    # Load picture
    img = Image.open(path)  # 1300*1800
    im = np.array(img, dtype=np.uint8)  # (1800, 1300, 4)
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Circle patch
    """
        Parameters
        ----------
        xy : (float, float)
            xy coordinates of ellipse centre.
        width : float
            Total length (diameter) of horizontal axis.
        height : float
            Total length (diameter) of vertical axis.
        angle : float, default: 0
            Rotation in degrees anti-clockwise.

        Notes
        -----
        Valid keyword arguments are:

        %(Patch)s
    """

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 36), 40, 50, 0, linewidth=0.1, edgecolor=[
                               200/255, 200/255, 169/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 36), 40, 50, 0, linewidth=0.1, edgecolor=[
                               249/255, 205/255, 173/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 96), 40, 50, 0, linewidth=0.1, edgecolor=[
                               131/255, 175/255, 151/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 96), 40, 50, 0, linewidth=0.1, edgecolor=[
                               182/255, 194/255, 154/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 156), 40, 50, 0, linewidth=0.1, edgecolor=[
                               178/255, 200/255, 187/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 156), 40, 50, 0, linewidth=0.1, edgecolor=[
                               201/255, 186/255, 131/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 216), 40, 50, 0, linewidth=a, edgecolor=[
                               160/255, 191/255, 124/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 216), 40, 50, 0, linewidth=0.1, edgecolor=[
                               179/255, 168/255, 150/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 286), 40, 50, 0, linewidth=a, edgecolor=[
                               113/255, 150/255, 159/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 286), 40, 50, 0, linewidth=a, edgecolor=[
                               225/255, 233/255, 220/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 346), 40, 50, 0, linewidth=a, edgecolor=[
                               204/255, 202/255, 169/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 346), 40, 50, 0, linewidth=a, edgecolor=[
                               227/255, 230/255, 195/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 406), 40, 50, 0, linewidth=a, edgecolor=[
                               219/255, 208/255, 167/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 406), 40, 50, 0, linewidth=a, edgecolor=[
                               179/255, 168/255, 150/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 466), 40, 50, 0, linewidth=a, edgecolor=[
                               205/255, 164/255, 158/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 466), 40, 50, 0, linewidth=a, edgecolor=[
                               210/255, 188/255, 167/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 526), 40, 50, 0, linewidth=a, edgecolor=[
                               189/255, 172/255, 156/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 526), 40, 50, 0, linewidth=a, edgecolor=[
                               199/255, 237/255, 233/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 586), 40, 50, 0, linewidth=a, edgecolor=[
                               243/255, 244/255, 246/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 586), 40, 50, 0, linewidth=a, edgecolor=[
                               196/255, 226/255, 216/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 646), 40, 50, 0, linewidth=a, edgecolor=[
                               166/255, 137/255, 124/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 646), 40, 50, 0, linewidth=a, edgecolor=[
                               219/255, 207/255, 202/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 706), 40, 50, 0, linewidth=a, edgecolor=[
                               205/255, 179/255, 128/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 706), 40, 50, 0, linewidth=a, edgecolor=[
                               170/255, 138/255, 87/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 766), 40, 50, 0, linewidth=a, edgecolor=[
                               114/255, 111/255, 128/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 766), 40, 50, 0, linewidth=a, edgecolor=[
                               178/255, 190/255, 126/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 826), 40, 50, 0, linewidth=a, edgecolor=[
                               113/255, 175/255, 164/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 826), 40, 50, 0, linewidth=a, edgecolor=[
                               229/255, 190/255, 157/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 886), 40, 50, 0, linewidth=a, edgecolor=[
                               101/255, 147/255, 74/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 886), 40, 50, 0, linewidth=a, edgecolor=[
                               160/255, 191/255, 124/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 946), 40, 50, 0, linewidth=a, edgecolor=[
                               205/255, 201/255, 125/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 946), 40, 50, 0, linewidth=a, edgecolor=[
                               229/255, 190/255, 157/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 1006), 40, 50, 0, linewidth=a, edgecolor=[
                               222/255, 156/255, 83/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 1006), 40, 50, 0, linewidth=a, edgecolor=[
                               222/255, 211/255, 140/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 1106), 40, 50, 0, linewidth=a, edgecolor=[
                               137/255, 190/255, 178/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 1106), 40, 50, 0, linewidth=a, edgecolor=[
                               201/255, 186/255, 131/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 1206), 40, 50, 0, linewidth=a, edgecolor=[
                               174/255, 221/255, 129/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 1206), 40, 50, 0, linewidth=a, edgecolor=[
                               138/255, 171/255, 202/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 1306), 40, 50, 0, linewidth=a, edgecolor=[
                               167/255, 220/255, 224/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 1306), 40, 50, 0, linewidth=a, edgecolor=[
                               178/255, 190/255, 126/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 1406), 40, 50, 0, linewidth=a, edgecolor=[
                               179/255, 214/255, 110/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 1406), 40, 50, 0, linewidth=a, edgecolor=[
                               96/255, 143/255, 159/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 1506), 40, 50, 0, linewidth=a, edgecolor=[
                               178/255, 200/255, 187/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 1506), 40, 50, 0, linewidth=a, edgecolor=[
                               158/255, 157/255, 131/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((50+transparent*50, 1606), 40, 50, 0, linewidth=a, edgecolor=[
                               173/255, 195/255, 192/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    for transparent in range(0, 10):
        # 控制透明度 十个等级
        rect = patches.Ellipse((600+transparent*50, 1606), 40, 50, 0, linewidth=a, edgecolor=[
                               230/255, 180/255, 80/255, transparent * 0.1], facecolor='none')
        ax.add_patch(rect)

    # 保存图片
    # 取消坐标轴显示
    plt.axis('off')
    plt.savefig('output/addEllipse2.jpg', dpi=2000, bbox_inches="tight")
    # plt.show()
    print('drawWaterMarkColorOfEllipse done')
    return


# 采用实心作图，探索可以通过扫描失去的水印
def tryWaterMask(path):
    # Load picture
    img = Image.open(path)  # 1300*1800
    im = np.array(img, dtype=np.uint8)
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    """
        Parameters
        ----------
        xy : (float, float)
            The anchor point.
        width : float
            Rectangle width.
        height : float
            Rectangle height.
        angle : float, default: 0
            Rotation in degrees anti-clockwise about *xy*.

        Other Parameters
        ----------------                        
        **kwargs : `.Patch` properties
            %(Patch)s
    """
    rect = patches.Rectangle(
        (100, 100), 200, 200,  facecolor=[201/255, 186/255, 131/255, 0.1])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (400, 100), 200, 200,  facecolor=[229/255, 190/255, 157/255, 0.1])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (700, 100), 200, 200,  facecolor=[222/255, 211/255, 140/255, 0.1])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (1000, 100), 200, 200,  facecolor=[179/255, 214/255, 110/255, 0.1])
    ax.add_patch(rect)

    # 透明度0.15
    rect = patches.Rectangle(
        (100, 400), 200, 200,  facecolor=[201/255, 186/255, 131/255, 0.15])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (400, 400), 200, 200,  facecolor=[229/255, 190/255, 157/255, 0.15])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (700, 400), 200, 200,  facecolor=[222/255, 211/255, 140/255, 0.15])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (1000, 400), 200, 200,  facecolor=[179/255, 214/255, 110/255, 0.15])
    ax.add_patch(rect)

    # 透明度0.05
    a = 700
    tm = 0.08
    rect = patches.Rectangle(
        (100, a), 200, 200,  facecolor=[201/255, 186/255, 131/255, tm])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (400, a), 200, 200,  facecolor=[229/255, 190/255, 157/255, tm])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (700, a), 200, 200,  facecolor=[222/255, 211/255, 140/255, tm])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (1000, a), 200, 200,  facecolor=[179/255, 214/255, 110/255, tm])
    ax.add_patch(rect)

    # 透明度0.05
    a = 1000
    tm = 0.08
    rect = patches.Rectangle(
        (100, a), 200, 200,  facecolor=[0.8, 0.8, 0.8, 0.08])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (400, a), 200, 200,  facecolor=[0.8, 0.8, 0.8, 0.1])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (700, a), 200, 200,  facecolor=[0.8, 0.8, 0.8, 0.12])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (1000, a), 200, 200,  facecolor=[0.8, 0.8, 0.8, 0.14])
    ax.add_patch(rect)

    a = 1300
    tm = 0.08
    rect = patches.Rectangle(
        (100, a), 200, 200,  facecolor=[0.5, 0.5, 0.5, 0.08])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (400, a), 200, 200,  facecolor=[0.5, 0.5, 0.5, 0.1])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (700, a), 200, 200,  facecolor=[0.5, 0.5, 0.5, 0.12])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (1000, a), 200, 200,  facecolor=[0.5, 0.5, 0.5, 0.14])
    ax.add_patch(rect)
    a = 1550
    tm = 0.08
    rect = patches.Rectangle(
        (100, a), 200, 200,  facecolor=[0.8, 0.8, 0.8, 0.08])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (400, a), 200, 200,  facecolor=[0.8, 0.8, 0.8, 0.1])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (700, a), 200, 200,  facecolor=[0.8, 0.8, 0.8, 0.12])
    ax.add_patch(rect)
    rect = patches.Rectangle(
        (1000, a), 200, 200,  facecolor=[0.8, 0.8, 0.8, 0.14])
    ax.add_patch(rect)

    # 保存图片
    # 取消坐标轴显示
    plt.axis('off')
    plt.savefig('output/waterMark.jpg', dpi=2000, bbox_inches="tight")

    print('done')


def writeLineMask2(lenX, lenY):
    # Load picture
    # img = Image.open(path)  # 500*511

    # print(img.size)
    # im = np.array(img, dtype=np.uint8)
    # print(type(im))

    im = np.ones((400, 600), dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    rect = patches.Rectangle((0, 0), 600, 400, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    # Create a Rectangle patch
    """
        Parameters
        ----------
        xy : (float, float)
            The anchor point.
        width : float
            Rectangle width.
        height : float
            Rectangle height.
        angle : float, default: 0
            Rotation in degrees anti-clockwise about *xy*.

        Other Parameters
        ----------------
        **kwargs : `.Patch` properties
            %(Patch)s
    """
    # 两重循环 探究线段影响
    y = 0
    while y < 400:
        r = random.randint(0, 10)
        x = 0+r
        while x < 600:
            rect = patches.Rectangle((x, y), lenX, lenY, linewidth=0.1, edgecolor=[
                                     230/255, 230/255, 230/255, 1], facecolor=[
                                     230/255, 230/255, 230/255, 1])
            ax.add_patch(rect)
            x = x+lenX+2+r
        y = y+lenY+1

    rect = patches.Rectangle((0, 0), 50, 400, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((150, 0), 100, 400, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((350, 0), 100, 400, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((590, 0), 10, 200, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((80, 0), 40, 150, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((80, 250), 40, 150, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)

    rect = patches.Rectangle((450, 100), 30, 300, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((570, 100), 30, 300, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)

    # 保存图片
    # 取消坐标轴显示
    plt.axis('off')
    name = 'output/hit4/hitwaterMark'+str(lenY)+'_'+str(lenX)+'.jpg'
    plt.savefig(name, dpi=1500, bbox_inches="tight")
    plt.close()
    print('done:', lenX, ',', lenY)


def writeLineMask3(lenX, lenY):
    # Load picture
    # img = Image.open(path)  # 500*511

    # print(img.size)
    # im = np.array(img, dtype=np.uint8)
    # print(type(im))

    im = np.ones((400, 600), dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    rect = patches.Rectangle((0, 0), 600, 400, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    # Create a Rectangle patch
    """
        Parameters
        ----------
        xy : (float, float)
            The anchor point.
        width : float
            Rectangle width.
        height : float
            Rectangle height.
        angle : float, default: 0
            Rotation in degrees anti-clockwise about *xy*.

        Other Parameters
        ----------------
        **kwargs : `.Patch` properties
            %(Patch)s
    """
    # 两重循环 探究线段影响
    y = 0
    while y < 400:
        r = random.randint(0, 10)
        x = 0+r
        while x < 600:
            rect = patches.Rectangle((x, y), lenX, 1, linewidth=0.1, edgecolor=[
                                     235/255, 235/255, 235/255, 1], facecolor=[
                                     235/255, 235/255, 235/255, 1])
            ax.add_patch(rect)
            x = x+lenX+2+r
        y = y+lenY+1

    rect = patches.Rectangle((0, 0), 50, 400, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((150, 0), 100, 400, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((350, 0), 100, 400, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((590, 0), 10, 200, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((80, 0), 40, 150, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((80, 250), 40, 150, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)

    rect = patches.Rectangle((450, 100), 30, 300, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)
    rect = patches.Rectangle((570, 100), 30, 300, linewidth=0.1, edgecolor=[
                             1, 1, 1, 1], facecolor=[1, 1, 1, 1])
    ax.add_patch(rect)

    # 保存图片
    # 取消坐标轴显示
    plt.axis('off')
    name = 'output/hit5/hitwaterMark'+str(lenY)+'_'+str(lenX)+'.jpg'
    plt.savefig(name, dpi=1500, bbox_inches="tight")
    plt.close()
    print('done:', lenX, ',', lenY)


if __name__ == '__main__':
    # test()
    lenY = 1
    while lenY < 10:
        lenX = 20
        while lenX < 50:

            writeLineMask2(lenX, lenY)

            writeLineMask3(lenX, lenY)
            lenX = lenX+5
        lenY = lenY+1

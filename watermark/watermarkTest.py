# -*- coding: utf-8 -*-

import cv2
import numpy


from PIL import Image, ImageDraw, ImageFont

# 在图片上添加水印


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


if __name__ == '__main__':
    str = input('input: ')
    print('your input is : ', str)

    addChaneseWaterMark(str)
    print('hi world!')

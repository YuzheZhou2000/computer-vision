# -*- coding: utf-8 -*-
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
inputPath = 'image/test/'
outputPath = 'output/mask/'


def writebaseMask():
    img = Image.open(outputPath+'base_mask'+'.png')
    img = img.convert("RGBA")
    width = img.size[0]
    high = img.size[1]
    detImg = cv2.imread('image/templet/fangwei.png')
    resize_img = cv2.resize(detImg, dsize=(width, high))
    masImage = Image.fromarray(cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB))
    for i in range(0, width):
        for j in range(0, high):
            data = masImage.getpixel((i, j))
            if (data.count(0) != 3):
                img.putpixel((i, j), (255, 255, 255, 0))
    img.save('2.png')


if __name__ == '__main__':
    writebaseMask()
    # mixByopencv('image/test/test-3.png', 'output/mask/triangle_mask.png', 3)

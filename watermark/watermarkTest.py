# -*- coding: utf-8 -*-

import cv2
import numpy


from PIL import Image, ImageDraw, ImageFont

# ��ͼƬ�����ˮӡ


def addTextWaterMark(text):
    # ������Ҫ���ˮӡ��ͼƬ
    img = cv2.imread('image/demo.jpg')

    # �������ˮӡ ������ԭʼͼ��-���ˮӡ���ַ���-����=����-�ֺ�-��ɫ-��ϸ
    img2 = cv2.putText(img, text, (100, 100),
                       cv2.LINE_AA, 2, (0, 249, 249), 4)

    # ����ͼƬ
    cv2.imwrite('output/waterMark.jpg', img2)

    # �ڴ�����ʾ
    cv2.imshow('img', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def addChaneseWaterMark(text):
    # ��ȡͼƬ��Ϣ
    img = cv2.imread('image/demo.jpg')
    # �ж��Ƿ���openCVͼƬ����
    if (isinstance(img, numpy.ndarray)):
        # ת����PIL����
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # ����ĸ�ʽ
    textSize = 80
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # �����ı�
    left = 100
    top = 100
    textColor = (168, 121, 103)
    draw.text((left, top), text, textColor, font=fontStyle)
    # ת����OpenCV����
    img2 = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    # ����ͼƬ
    cv2.imwrite('output/wj.jpg', img2)
    cv2.imshow('img', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    str = input('input: ')
    print('your input is : ', str)

    addChaneseWaterMark(str)
    print('hi world!')

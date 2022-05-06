import cv2
import numpy as np
import matplotlib.pyplot as plt


def test():
    # 200*200
    image = cv2.imread("data/iu-lizhien2.jpg")

    # 显示图像
    # 存在色差
    # 原因：opencv的颜色通道顺序为[B,G,R]
    # 而matplotlib的颜色通道顺序为[R,G,B]
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    # 调整后
    # image2 = image[:,:,(2,1,0)]
    # plt.imshow(image2)
    # plt.axis('off')
    # plt.show()

    """
    图像灰度化算法
    Gray = 0.299R+0.587G+0.114*B
    """
    r, g, b = [image[:, :, i] for i in range(3)]
    img_gray = r * 0.299 + g * 0.587 + b * 0.114
    plt.imshow(img_gray, cmap="gray")
    plt.axis('off')
    plt.show()


def eaualHist_demo(image):
    # opencv的直方图均衡化要基于单通道灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.namedWindow('orgin_image', cv2.WINDOW_NORMAL)
    cv2.imshow('origin_image', gray)

    # 自动调整图像对比度，把图像变得更清晰
    dst = cv2.equalizeHist(gray)
    cv2.namedWindow('equalize_image', cv2.WINDOW_NORMAL)
    cv2.imshow("equalize_image", dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scan_find():
    scan_image = cv2.imread("data2/scan.jpg")  # 460*460
    print_image = cv2.imread("data2/print.jpg")  # 460*460
    eaualHist_demo(scan_image)


if __name__ == '__main__':
    scan_find()

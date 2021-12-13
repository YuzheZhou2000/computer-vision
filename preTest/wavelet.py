import cv2
import numpy as np
from pywt import dwt2, idwt2

"""

二维小波变换(一维和n维类似)：

# 单层变换 pywt.dwt2

pywt.dwt2(data, wavelet, mode=’symmetric’, axes=(-2, -1))
data: 输入的数据
wavelet:小波基
mode: 默认是对称的
return: (cA, (cH, cV, cD))要注意返回的值，分别为低频分量，水平高频、垂直高频、对角线高频。高频的值包含在一个tuple中。


#单层逆变换 pywt.idwt2

pywt.idwt2(coeffs, wavelet, mode, axes)
coeffs: 经小波变换后得到的各层的系数
wavelet:小波基

# 多尺度变换 wavedec2

pywt.wavedec2(data, wavelet, mode=’symmetric’, level=None, axes=(-2, -1))
data: 输入的数据
wavelet:小波基
level: 尺度（要变换多少层）
return： 返回的值要注意，每一层的高频都是包含在一个tuple中，例如三层的话返回为 [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2)， (cH1, cV1, cD1)]


"""

# 读取灰度图
image = cv2.imread('data/iu-lizhien.jpeg', 0)

print(type(image))
print(len(image),len(image[0]))
print(image[0])

# 进行haar 小波变换
cA, (cH, cV, cD) = dwt2(image, 'haar')
print(type(image))
print(len(cA),len(cA[0]))
print(cA[0])
# 小波变换之后，低频分量对应的图像：
cv2.imwrite('lena.png', np.uint8(cA / np.max(cA) * 255))

# # 小波变换之后，水平方向高频分量对应的图像：
# cv2.imwrite('lena_h.png', np.uint8(cH / np.max(cH) * 255))
# # 小波变换之后，垂直平方向高频分量对应的图像：
# cv2.imwrite('lena_v.png', np.uint8(cV / np.max(cV) * 255))
# # 小波变换之后，对角线方向高频分量对应的图像：
# cv2.imwrite('lena_d.png', np.uint8(cD / np.max(cD) * 255))
#
# # 根据小波系数重构回去的图像
# rimg = idwt2((cA, (cH, cV, cD)), 'haar')
# cv2.imwrite("reimage.png", rimg)

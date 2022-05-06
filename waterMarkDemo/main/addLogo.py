from typing import Type
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
inputPath = 'image/test/'
outputPath = 'output/mask/'
templetPath = 'image/templet/'
len_X = 40
len_Y = 2


class generateWatermark:
    def writebaseMask(self, lenX, lenY, shape):
        if os.path.isfile(outputPath+shape+'_mask.png'):
            print('log :  writebaseMask already done!  ')
            return Image.open(outputPath+shape+'_mask.png')
        im = np.ones((400, 600), dtype=np.uint8)
        # Create figure and axes
        fig, ax = plt.subplots(1)
        # Display the image
        ax.imshow(im)
        rect = patches.Rectangle((0, 0), 600, 400, linewidth=0.1, edgecolor=[
            1, 1, 1, 1], facecolor=[1, 1, 1, 1])
        ax.add_patch(rect)
        y = 0
        while y < 400:
            r = random.randint(0, 10)
            x = 0+r
            while x < 600:
                rect = patches.Rectangle((x, y), lenX, 1, linewidth=0.1, edgecolor=[
                    240/255, 240/255, 240/255, 1], facecolor=[
                    240/255, 240/255, 240/255, 1])
                ax.add_patch(rect)
                x = x+lenX+2+r
            y = y+lenY+1
        plt.axis('off')
        name = outputPath+'base_mask'+'.png'
        plt.savefig(name, dpi=1500, bbox_inches="tight", pad_inches=0.0)
        img = Image.open(outputPath+'base_mask'+'.png')

        img = img.convert("RGBA")
        width = img.size[0]
        high = img.size[1]
        detImg = cv2.imread(templetPath+shape + '.png')
        print(templetPath+shape + '.png')
        resize_img = cv2.resize(detImg, dsize=(width, high))
        masImage = Image.fromarray(
            cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB))
        for i in range(0, width):
            for j in range(0, high):
                data = masImage.getpixel((i, j))
                if (data.count(0) != 3):
                    img.putpixel((i, j), (255, 255, 255, 0))

        # if shape == 'triangle':
        #     img = img.convert("RGBA")
        #     width = img.size[0]
        #     high = img.size[1]
        #     for i in range(0, high):
        #         for j in range(0, i):
        #             img.putpixel((i, j), (255, 255, 255, 0))

        #     for i in range(high, width-1):
        #         for j in range(0, high-1):
        #             img.putpixel((i, j), (255, 255, 255, 0))
        # if shape == 'fangwei':
        #     img = img.convert("RGBA")
        #     width = img.size[0]
        #     high = img.size[1]
        #     detImg = cv2.imread('image/templet/fangwei.png')
        #     resize_img = cv2.resize(detImg, dsize=(width, high))
        #     masImage = Image.fromarray(
        #         cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB))
        #     for i in range(0, width):
        #         for j in range(0, high):
        #             data = masImage.getpixel((i, j))
        #             if (data.count(0) != 3):
        #                 img.putpixel((i, j), (255, 255, 255, 0))
        # if shape == 'hit':
        #     img = img.convert("RGBA")
        #     width = img.size[0]
        #     high = img.size[1]
        #     detImg = cv2.imread('image/templet/hit.png')
        #     resize_img = cv2.resize(detImg, dsize=(width, high))
        #     masImage = Image.fromarray(
        #         cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB))
        #     for i in range(0, width):
        #         for j in range(0, high):
        #             data = masImage.getpixel((i, j))
        #             if (data.count(0) != 3):
        #                 img.putpixel((i, j), (255, 255, 255, 0))

        img.save(outputPath+shape + '_mask.png')
        return img


class processPhoto:
    # init the class
    def __init__(self, orgPath, maskPath):
        self.orgPath = orgPath
        self.maskPath = maskPath

    # getSizeOfPhoto
    def getSizeOfPhoto(self, arg):
        if arg == 'org':
            path = self.orgPath
            # open image
            img = Image.open(path)
            self.orgSize = img.size
            img.close()
        elif arg == 'mask':
            path = self.maskPath
            img = Image.open(path)
            self.maskSize = img.size
            img.close()

    # add logo
    def addLogo(self):

        self.getSizeOfPhoto('org')
        layer = Image.new('RGBA', self.orgSize)
        # add logo to photo
        layer.paste(Image.open(self.maskPath), [0, 0])

        # layer.save(outputPath+'zhou.png') to opencv
        img = cv2.cvtColor(np.asarray(layer), cv2.COLOR_RGB2BGR)
        rows, cols, channels = img.shape
        roi = img[0:rows, 0:cols]

    # transparency_Image
    def transPNG(self, name):
        srcImageName = inputPath+name
        img = Image.open(srcImageName)
        img = img.convert("RGBA")
        size1 = img.size[0]
        size2 = img.size[1]

        # resize the image

        datas = img.getdata()

        width, height = img.size
        for i in range(0, width):
            for j in range(0, height):
                data = img.getpixel((i, j))
                if (data.count(255) == 4):
                    img.putpixel((i, j), (255, 255, 255, 0))

        return img

    # fix_Image
    def mix(self, imageOrg, imageMask, coordinator):
        '''
        param:  
            imageOrg     : the photo of docume
            imageMask    : our waterMask
            coordinator  : The coordinates of the watermark
        '''
        im = imageMask
        mark = imageOrg
        layer = Image.new('RGBA', mark.size, (0, 0, 0, 0))
        layer.paste(mark, coordinator)
        out = Image.composite(layer, im, layer)
        return out

    def mixByopencv(self, density):
        # import source image
        sorImg = cv2.imread(self.orgPath)
        high = sorImg.shape[0]
        width = sorImg.shape[1]

        # make a white photo
        I = np.zeros((high, width), dtype=np.uint8)
        I = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
        I[:, :, 0] = 255
        I[:, :, 1] = 255
        I[:, :, 2] = 255

        # import mark
        detImg = cv2.imread(self.maskPath)

        # rezie mark
        rezie_width = int(width/density)
        rezie_high = int((rezie_width * high)/width)
        resize_img = cv2.resize(detImg, dsize=(rezie_width, rezie_high))

        # add mark
        i = 0
        while (i+1)*rezie_high < high:
            for j in range(0, density):
                I[i * rezie_high:(i+1)*rezie_high, j *
                  rezie_width:(j+1)*rezie_width] = resize_img
            i = i+1

        cv2.imwrite(outputPath + 'baseMask.png', I)

        # add image
        sorImage = Image.fromarray(cv2.cvtColor(sorImg, cv2.COLOR_BGR2RGB))
        masImage = Image.fromarray(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

        width, height = masImage.size
        for i in range(0, width):
            for j in range(0, height):
                data = sorImage.getpixel((i, j))
                if (data.count(255) == 4):
                    masImage.putpixel((i, j), data)

        srcImageName = inputPath + 'test-3.png'
        desImageNmae = outputPath + 'baseMask.png'

        img = Image.open(srcImageName)
        img = img.convert("RGBA")

        img1 = Image.open(desImageNmae)
        img1 = img1.convert("RGBA")

        width, height = img.size
        for i in range(0, width):
            for j in range(0, height):
                data = img.getpixel((i, j))

                if (data.count(255) == 4):
                    img.putpixel((i, j), img1.getpixel((i, j)))

        img.save(outputPath + 'waterMask.png')
        return img

    # save_Image

    def saveImage(self, imageOut, name):
        imageOut.save(outputPath+name+'.png')
        print('log :  ' + outputPath+name+'.png' + '  save done')


if __name__ == '__main__':

    shape = input('input the mask shape: ')  # shape = 'triangle'
    imgName = input('input the image name: ')  # imgName = 'test-3.png'
    density = input('input the mask density: ')  # 3

    density = int(density)

    # step 1 : Generate watermark
    print("step 1 : Generate watermark...")
    imageMask = generateWatermark().writebaseMask(len_X, len_Y, shape)

    # step 2 : Initialization processing logic
    print("step 2 : Initialization processing logic...")
    p2p = processPhoto(inputPath+imgName, outputPath+shape + '_mask'+'.png')

    # step 3 : Image transparency
    print("step 3 : Image transparency...")
    imageTrans = p2p.transPNG(imgName)

    # step 4 : Fix Image and Mask
    print("step 4 : Fix Image and Mask...")
    outImage = p2p.mixByopencv(density)

    # step 5 : save the image
    print("step 5 : save the image...")
    p2p.saveImage(outImage, 'finnaly')

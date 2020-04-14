import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


# 使用unger平滑处理去除孤立的噪点
def ungerSmooth(image):
    w = image.shape[1]
    h = image.shape[0]
    if w<3 or h<3:
        return image
    for x in range(1,h-1):
        for y in range(1,w-1):
            if image[x][y]==1:
                isChange = (image[x][y+1] + image[x-1][y+1]+image[x-1][y])*(image[x+1][y]+image[x+1][y-1]+image[x][y-1]) + (image[x-1][y]+image[x-1][y-1]+image[x][y-1])*(image[x+1][y]+image[x+1][y+1]+image[x][y+1])
                if isChange==0:
                    image[x][y] = 0
            else:
                isChange = image[x-1][y]*image[x+1][y]*(image[x][y-1]+image[x][y+1])+image[x][y-1]*image[x][y+1]*(image[x-1][y]+image[x+1][y])
                if isChange!=0:
                    image[x][y] = 1
    return image

# 获取字符最小内接矩阵函数
def getMinRect(symbols):
    minRects = []
    for symbol in symbols:
        symbol = np.array(symbol)
        # y = np.min(symbol[:,0])
        # x = np.min(symbol[:,1])
        # h = np.max(symbol[:,0])-np.min(symbol[:,0])+1
        # w = np.max(symbol[:,1])-np.min(symbol[:,1])+1
        x, y, w, h = cv2.boundingRect(symbol)
        minRects.append((y,x,h,w))
    return minRects

# 缩放到指定大小的二值图片，符号居中
def scaleImg(imgSize,extracted_img,w,h):
    symbol_img = np.zeros((imgSize, imgSize))
    if (w > h):
        if (int)(h * imgSize / w) == 0:
            res_h = 1
        else:
            res_h = (int)(h * imgSize / w)
        res = cv2.resize(extracted_img, (imgSize, res_h),
                         interpolation=cv2.INTER_NEAREST)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        symbol_img[d:res.shape[0] + d, 0:res.shape[1]] = res
    else:
        if (int)(w * imgSize / h) == 0:
            res_w = 1
        else:
            res_w = (int)(w * imgSize / h)
        res = cv2.resize(extracted_img, (res_w, imgSize),
                         interpolation=cv2.INTER_NEAREST)
        d = int(abs(res.shape[0] - res.shape[1]) / 2)
        symbol_img[0:res.shape[0], d:res.shape[1] + d] = res
    return symbol_img


# 将符号转换为指定大小的照片
def getSymbolImg(symbol,minRect,imgSize,binary_img):
    blank_img = np.zeros(binary_img.shape)
    tmp_symbol = tuple(np.array(symbol).T)
    print("minRects[i]:", minRect)
    x, y, w, h = minRect
    blank_img[tmp_symbol] = 1
    extracted_img = blank_img[y:y + h, x:x + w]
    return scaleImg(imgSize,extracted_img,w,h)

# 获取二值化后的图像
def getBinaryImg(ori_image,h=22,sw=5,sh=11):
    # print(ori_image.shape)
    # 将彩色图转化成灰度图
    img_gray = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    # NLM算法进行灰度图降噪
    img_gray = cv2.fastNlMeansDenoising(img_gray, None, h, sw, sh)
    # 自定义卷积核
    kernel_sharpen_1 = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]])
    blur = cv2.filter2D(img_gray, -1, kernel_sharpen_1)
    # plt.imshow(blur, cmap="Greys", interpolation="None")
    # plt.show()
    ret, binary_img = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_img
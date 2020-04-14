import cv2
import numpy as np
import matplotlib.pyplot as plt

def VThin(image, array):
    h = image.shape[0]
    w = image.shape[1]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j - 1] + image[i, j] + image[i, j + 1] if 0 < j < w - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l]==255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def HThin(image, array):
    h = image.shape[0]
    w = image.shape[1]
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i - 1, j] + image[i, j] + image[i + 1, j] if 0 < i < h - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l]==255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image

def Two(image):
    iTwo = np.zeros(image.shape,dtype=int)
    for i in range(iTwo.shape[0]):
        for j in range(iTwo.shape[1]):
            iTwo[i,j] = 0 if image[i,j] < 200 else 255
    return iTwo

def Xihua(image, array, num=10):
    iXihua = image.copy()
    iXihua[iXihua==0] = 255
    iXihua[iXihua==1] = 0
    for i in range(num):
        VThin(iXihua, array)
        HThin(iXihua, array)
    iXihua[iXihua==0] = 1
    iXihua[iXihua==255] = 0
    return iXihua


if __name__ == "__main__":

    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

    image = cv2.imread('C:/Users/Pengr/Desktop/Test3.jpg')
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, iTwo = cv2.threshold(img_gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # iTwo = Two(img_gray)
    iThin = Xihua(iTwo, array)
    plt.imshow(image, cmap="Greys", interpolation="None")
    plt.show()
    plt.imshow(iTwo, cmap="Greys", interpolation="None")
    plt.show()
    plt.imshow(iThin, cmap="Greys", interpolation="None")
    plt.show()
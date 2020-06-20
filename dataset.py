import os
import cv2
import json
import global_config
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from img_process import getBinaryImg

class Dataset():
    # 图像生成为正方形
    def __init__(self,dataCount=0,size=100,filename="./data/symbols.data",originalPath="./data/allShapes"):
        self.filename = filename
        self.imgSize = size
        self.originalPath = originalPath
        self.dataCount = dataCount
        self.originalShapes = global_config.ALL_SHAPES

        self.currTrainLen = 0
        self.currTestLen = 0
        self.createImgData()
        self.trainData,self.testData = self.loadData()
        self.trainDataLen = self.trainData["image"].shape[0]
        self.testDataLen = self.testData["image"].shape[0]

    def createImgData(self):
        if self.dataCount==0:
            return
        print("-----开始生成原始数据-----")
        data = []
        for shape in self.originalShapes:
            if shape!="z":
                continue
            if len(shape) == 1 and ord(shape) >= 65 and ord(shape) <= 90:
                shape_name = shape + "_upper"
            elif shape=="i" or shape=="j":
                shape_name = shape+"_low_vl"
            else:
                shape_name = shape
            shapeDirPath = os.path.join(self.originalPath, shape_name)

            for maindir, subdir, file_name_list in os.walk(shapeDirPath):
                for file_name in file_name_list:
                    file_path = os.path.join(maindir, file_name)
                    oriImg = cv2.imread(file_path)

                    for i in range(self.dataCount):
                        h,sw,sh = np.random.randint(5,40,(3,))
                        binaryImg = getBinaryImg(oriImg,h,sw,sh)
                        # 将二值图中值为1的像素点取出，既图像中的字符
                        binaryImg = skeletonize(binaryImg).astype(np.int32)  # skeletonize函数用于将图像轮廓转为宽度为1个像素的图像
                        symbol = np.where(binaryImg != 0)
                        tmp_symbol = np.array(symbol).T.reshape(np.array(symbol).T.shape[0], 1, 2)
                        tmp = tmp_symbol[:, :, 0].copy()
                        tmp_symbol[:, :, 0] = tmp_symbol[:, :, 1]
                        tmp_symbol[:, :, 1] = tmp
                        x,y,w,h = cv2.boundingRect(tmp_symbol)
                        extracted_img = binaryImg[y:y + h, x:x + w]
                        symbol_img = np.zeros((self.imgSize,self.imgSize))
                        if (w > h):
                            res = cv2.resize(extracted_img, (self.imgSize, (int)(h * self.imgSize / w)),
                                             interpolation=cv2.INTER_NEAREST)
                            d = int(abs(res.shape[0] - res.shape[1]) / 2)
                            symbol_img[d:res.shape[0] + d, 0:res.shape[1]] = res
                        else:
                            res = cv2.resize(extracted_img, ((int)(w * self.imgSize / h), self.imgSize),
                                             interpolation=cv2.INTER_NEAREST)
                            d = int(abs(res.shape[0] - res.shape[1]) / 2)
                            symbol_img[0:res.shape[0], d:res.shape[1] + d] = res
                        # plt.imshow(res,cmap="Greys", interpolation="None")
                        # plt.show()
                        symbol_img = symbol_img*0.99
                        symbol_img = symbol_img.reshape((self.imgSize*self.imgSize,))
                        # print(res)
                        tmpdata = {}
                        tmpdata["image"] = symbol_img.tolist()
                        idx = self.originalShapes.index(shape)
                        # 将标签转为独热码
                        oneHot = np.zeros(len(self.originalShapes))
                        oneHot[idx] = 1
                        tmpdata["lable"] = oneHot.tolist()
                        data.append(tmpdata)
                        # print("count:",i)
                    print("original image:",file_name)
            print("符号:",shape,"生成完成-----")
        strData = json.dumps(data)
        file = open(self.filename, "w")
        file.write(strData)
        file.close()
        print("-----数据生成结束-----")

    def loadData(self,trainRate=0.7):
        print("-----开始载入数据-----")
        file = open(self.filename, "r")
        datas = np.array(json.loads(file.read()))
        idxes = np.arange(datas.shape[0])
        # replace值为False时则为不重复采样
        trainDataIdx = np.random.choice(idxes,int(datas.shape[0]*trainRate),replace=False)
        testDataIdx = np.setdiff1d(idxes,trainDataIdx)
        tmpTrainData = np.array(datas[trainDataIdx])
        tmpTestData = np.array(datas[testDataIdx])
        trainData = {}
        trainData["image"] = np.array([img["image"] for img in tmpTrainData]*20)
        trainData["lable"] = np.array([img["lable"] for img in tmpTrainData]*20)
        # trainData["image"][np.where(trainData["image"]==1)] = 0.99
        testData = {}
        testData["image"] = np.array([img["image"] for img in tmpTestData]*20)
        testData["lable"] = np.array([img["lable"] for img in tmpTestData]*20)
        # testData["image"][np.where(testData["image"] == 1)] = 0.99
        file.close()
        print("-----数据载入完成-----")
        return trainData,testData

    def nextTrainBatch(self,batchSize):
        if self.currTrainLen==self.trainData["image"].shape[0]:
            return

        batch_xs = self.trainData["image"][self.currTrainLen:self.currTrainLen+batchSize]
        batch_ys = self.trainData["lable"][self.currTrainLen:self.currTrainLen+batchSize]
        self.currTrainLen+=batchSize
        return batch_xs,batch_ys

    def nextTestBatch(self,batchSize):
        if self.currTestLen==self.testData["image"].shape[0]:
            return

        batch_xs = self.testData["image"][self.currTestLen:self.currTestLen+batchSize]
        batch_ys = self.testData["lable"][self.currTestLen:self.currTestLen+batchSize]
        self.currTestLen+=batchSize
        return batch_xs,batch_ys


if __name__ == "__main__":
    dataset = Dataset(dataCount=20,filename="./data/symbols5_5_z.data")
    for i in range(100):
        sym1 = np.array(dataset.trainData["image"][100*i]).reshape((dataset.imgSize,dataset.imgSize))
        lab1 = global_config.INFTYCDB_3_SHAPES[np.argmax(dataset.trainData["lable"][100*i])]
        print(lab1)
        plt.imshow(sym1,cmap="Greys", interpolation="None")
        plt.show()
    # print("trainData_lable_shape:",dataset.trainData["lable"].shape)
    # print("testData_lable_shape:",dataset.testData["lable"].shape)
    # for i in range(5):
    #     nextB_xs,nextB_ys = dataset.nextTrainBatch(10)
    #     print("nextB_lable:",np.argmax(nextB_ys[0]))


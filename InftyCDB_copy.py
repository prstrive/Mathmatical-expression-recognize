import os
import cv2
import json
import img_process
import global_config
import numpy as np
import pandas as pd
import global_config
import matplotlib.pyplot as plt

class InftyCDB():
    def __init__(self,filename,imgSize=global_config.IMG_SIZE):
        self.imgSize = imgSize
        self.filename = filename
        self.oriImgPath = "./data/InftyCDB-3/InftyCDB-3-A/images/"
        self.ocrCodeListPath = "./data/InftyCDB-3/OcrCodeList.txt"
        self.charInfoDB_3_A_InfoPath = "./data/InftyCDB-3/InftyCDB-3-A/CharInfoDB-3-A_Info.csv"
        self.originalShapes = global_config.INFTYCDB_3_SHAPES
        # self.currLen = 0
        # self.dataLen = 23500
        # self.createImgData()
        self.currTrainLen = 0
        self.currTestLen = 0
        self.trainData, self.testData = self.loadData()
        self.trainDataLen = self.trainData["image"].shape[0]
        self.testDataLen = self.testData["image"].shape[0]

    def createImgData(self):
        print("-----开始创建数据-----")
        ocrCodeList = pd.read_table(self.ocrCodeListPath, sep=",", names=["code", "type", "name"])
        charInfoDB_3_A_Info = pd.read_csv(self.charInfoDB_3_A_InfoPath)
        ocrCodeList["code"] = ocrCodeList["code"].map(lambda x:x[2:])

        file = open(self.filename, "a")
        for shape in self.originalShapes:
            if shape=="sqrt":
                for maindir, subdir, file_name_list in os.walk("./data/allShapes/sqrt"):
                    for file_name in file_name_list:
                        file_path = os.path.join(maindir, file_name)
                        oriImg = cv2.imread(file_path)
                        h, sw, sh = np.random.randint(5, 40, (3,))
                        binaryImg = img_process.getBinaryImg(oriImg, h, sw, sh)
                        # 将二值图中值为1的像素点取出，既图像中的字符
                        symbol = np.where(binaryImg != 0)
                        tmp_symbol = np.array(symbol).T.reshape(np.array(symbol).T.shape[0], 1, 2)
                        tmp = tmp_symbol[:, :, 0].copy()
                        tmp_symbol[:, :, 0] = tmp_symbol[:, :, 1]
                        tmp_symbol[:, :, 1] = tmp
                        x, y, w, h = cv2.boundingRect(tmp_symbol)
                        extracted_img = binaryImg[y:y + h, x:x + w]
                        extractImg_binary = img_process.scaleImg(self.imgSize,extracted_img, w, h)*0.99
                        imgInfo = {}
                        imgInfo["image"] = extractImg_binary.reshape((self.imgSize * self.imgSize,)).tolist()
                        oneHot = np.zeros(len(self.originalShapes))
                        oneHot[self.originalShapes.index(shape)] = 1
                        for i in range(200):
                            imgInfo["lable"] = oneHot.tolist()
                            strData = json.dumps(imgInfo)
                            file.write(strData + '\n')
            else:
                shape_codes = ocrCodeList[(ocrCodeList["name"]==shape) & (ocrCodeList["type"]!="Calligraphic") & (ocrCodeList["type"]!="German") & (ocrCodeList["type"]!="Script")]["code"].values
                shape_infos = pd.DataFrame(columns=charInfoDB_3_A_Info.columns)
                for code in shape_codes:
                    shape_infos = shape_infos.append(charInfoDB_3_A_Info[charInfoDB_3_A_Info["code"]==code])
                if shape_infos.shape[0]<=0:
                    print("当前形状: "+shape+" 没有找到数据")
                    continue
                for idx in shape_infos.index:
                    oriImg = cv2.imread(self.oriImgPath+str(shape_infos.loc[idx]["sheet"])+".png")
                    x = shape_infos.loc[idx]["cx"]
                    y = shape_infos.loc[idx]["cy"]
                    w = shape_infos.loc[idx]["width"]
                    h = shape_infos.loc[idx]["height"]
                    extractImg = oriImg[y:y+h,x:x+w]
                    extractImg_binary = img_process.scaleImg(self.imgSize,img_process.getBinaryImg(extractImg,h=15,sw=5,sh=11),w,h)*0.99

                    # print(extractImg_binary)
                    # plt.imshow(extractImg_binary,interpolation="None")
                    # plt.show()

                    imgInfo = {}
                    imgInfo["image"] = extractImg_binary.reshape((self.imgSize*self.imgSize,)).tolist()
                    oneHot = np.zeros(len(self.originalShapes))
                    oneHot[self.originalShapes.index(shape)] = 1
                    imgInfo["lable"] = oneHot.tolist()
                    strData = json.dumps(imgInfo)
                    file.write(strData + '\n')
                    # print("shape "+shape+" idx:",idx)
            print(shape+"录入完成")
        file.close()
        print("-----数据创建结束-----")

    def loadData(self,trainRate=0.7):
        print("-----开始载入数据-----")

        datas = []
        for line in open(self.filename,"r"):
            data = json.loads(line.strip("\n"))
            datas.append(data)
        datas = np.array(datas)
        datasIdxes = np.arange(datas.shape[0])
        trainIdx = np.random.choice(datasIdxes,int(datas.shape[0]*trainRate),replace=False)
        testIdx = np.setdiff1d(datasIdxes,trainIdx)
        tmp_train = datas[trainIdx]
        trainData = {}
        trainData["image"] = np.array([tmp["image"] for tmp in tmp_train])
        trainData["lable"] = np.array([tmp["lable"] for tmp in tmp_train])
        tmp_test = datas[testIdx]
        testData = {}
        testData["image"] = np.array([tmp["image"] for tmp in tmp_test])
        testData["lable"] = np.array([tmp["lable"] for tmp in tmp_test])

        print("-----数据载入完成-----")
        return trainData,testData

    def nextTrainBatch(self, batchSize):

        # if self.currTrainLen==0:
        #     self.trainFile = open(self.filename.split(".")[0] + "_train.data", "r")
        #     self.trainFile.seek(0)
        #
        # train_datas = []
        # for i in range(batchSize):
        #     data = json.loads(self.trainFile.readline().strip("\n"))
        #     train_datas.append(data)
        # train_datas = np.array(train_datas)
        # batch_xs = np.array([tmp["image"] for tmp in train_datas])
        # batch_ys = np.array([tmp["lable"] for tmp in train_datas])
        # self.currTrainLen += batchSize
        #
        # if self.currTrainLen >= self.trainDataLen:
        #     self.trainFile.close()
        batch_xs = self.trainData["image"][self.currTrainLen:self.currTrainLen+batchSize]
        batch_ys = self.trainData["lable"][self.currTrainLen:self.currTrainLen+batchSize]
        self.currTrainLen += batchSize
        return batch_xs, batch_ys

    def nextTestBatch(self, batchSize):

        # if self.currTestLen==0:
        #     self.testFile = open(self.filename.split(".")[0] + "_test.data", "r")
        #     self.trainFile.seek(0)
        #
        # test_datas = []
        # for i in range(batchSize):
        #     data = json.loads(self.testFile.readline().strip("\n"))
        #     test_datas.append(data)
        # test_datas = np.array(test_datas)
        # batch_xs = np.array([tmp["image"] for tmp in test_datas])
        # batch_ys = np.array([tmp["lable"] for tmp in test_datas])
        # self.currTestLen += batchSize
        #
        # if self.currTestLen>=self.testDataLen:
        #     self.testFlie.close()
        batch_xs = self.testData["image"][self.currTestLen:self.currTestLen + batchSize]
        batch_ys = self.testData["lable"][self.currTestLen:self.currTestLen + batchSize]
        self.currTestLen += batchSize
        return batch_xs, batch_ys

    # def __del__(self):
    #     self.trainFile.close()
    #     self.testFile.close()


if __name__ == "__main__":
    datas = InftyCDB(filename="./data/symbols2_test.data")
    # datas.createImgData()

    # datas.loadData()
    # print("datas.trainDataLen:",datas.trainDataLen)
    # print("datas.testDataLen:",datas.testDataLen)
    for i in range(datas.dataLen//200):
        datas.currTrainLen = 0
        tx,ty,ex,ey = datas.loadData(200)
        print(ty.shape,tx.shape,ex.shape,ey.shape)
        img = tx[0].reshape((global_config.IMG_SIZE,global_config.IMG_SIZE))
        plt.imshow(img, cmap="Greys", interpolation="None")
        plt.show()

    # for line in open("./data/symbols3_train.data","r"):
    #     line = line.strip("\n")  # 删除数据中的换行符
    #     data = json.loads(line)
    #     img = np.array(data["image"]).reshape((global_config.IMG_SIZE,global_config.IMG_SIZE))
    #     plt.imshow(img, cmap="Greys", interpolation="None")
    #     plt.show()
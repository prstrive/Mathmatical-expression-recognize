import graph_segmentation as graphSeg
import structure_ana as structureAna
import img_process
import cv2
import global_config
import matplotlib.pyplot as plt

imgPath = "your test image's path"

ori_image = cv2.imread(imgPath)
binary_img = img_process.getBinaryImg(ori_image)
unknownSymbols,knownSymbolsInfo,points = graphSeg.getSegmentation(binary_img,ori_image)
# 排好序的从左到右的字符
allSymbolsInfo = structureAna.recogSymbols(unknownSymbols,knownSymbolsInfo,binary_img)
# for sym in allSymbolsInfo:
#     sym_image = img_process.getSymbolImg(sym["symPixel"], sym["minRect"], global_config.IMG_SIZE,binary_img)
#     plt.imshow(sym_image, cmap="Greys", interpolation="None")
#     plt.show()

mes = structureAna.getMEs(allSymbolsInfo,points)

print("公式识别结果为:")
for me in mes:
    print(me)

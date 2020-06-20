# from dataset import  Dataset
# from global_variable import symClasser
# import matplotlib.pyplot as plt
# import numpy as np
# import global_config

# dataset = Dataset(dataCount=0,filename="./data/symbols5_3.data")
# for i in range(10):
#     sym1 = np.array(dataset.trainData["image"][500*i])
#     syns, ps = symClasser.predict([sym1])
#     print("candidata:", syns, "p:", ps)
#     lab1 = global_config.INFTYCDB_3_SHAPES[np.argmax(dataset.trainData["lable"][500*i])]
#     print(lab1)
#     sym1 = sym1.reshape((dataset.imgSize, dataset.imgSize))
#     plt.imshow(sym1,cmap="Greys", interpolation="None")
#     plt.show()

points = None

def tt(p):
    global points
    points = p

def tt2():
    print(points)

tt(10)
tt2()
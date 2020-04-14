import json
i = 0
datas = []
for line in open("E:/PyCharm/MER_2/data/symbols3.data","r"):
    data = json.loads(line.strip("\n"))
    datas.append(data)
    i+=1
    if i%20000==0:
        outFile = open("E:/PyCharm/MER_2/data/symbols3_" + str(i//20000) + ".data", "w")
        outFile.write(json.dumps(datas))
        datas = []
        outFile.close()
        print("文件 E:/PyCharm/MER_2/data/symbols3_" + str(i//20000)+ ".data 生成完毕")
if len(datas)>10000:
    outFile = open("E:/PyCharm/MER_2/data/symbols3_" + str(i // 20000) + ".data", "a")
    outFile.write(json.dumps(datas))
    datas = []
    outFile.close()
print("文件写入完成")
print("文件长度为：",i)

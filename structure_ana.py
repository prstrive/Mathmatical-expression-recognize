import img_process
import copy
import numpy as np
import global_config
import cv2
import matplotlib.pyplot as plt
from global_variable import symClasser

# 孤立的点字符
points = None
glo_binary_img = None

# 识别出符号，且保留可能的候选者
def recogSymbols(unknownSymbols,knownSymbolsInfo,binary_img):
    global glo_binary_img
    glo_binary_img = binary_img
    allSymbolsInfo = knownSymbolsInfo
    unknownSymbolsMinRects = img_process.getMinRect(unknownSymbols)
    for i in range(len(unknownSymbols)):
        sym_image = img_process.getSymbolImg(unknownSymbols[i], unknownSymbolsMinRects[i], global_config.IMG_SIZE, binary_img)*0.99
        # print(sym_image.tolist())
        # plt.imshow(sym_image, cmap="Greys", interpolation="None")
        # plt.show()
        sym_image = sym_image.reshape((sym_image.shape[0] * sym_image.shape[1],))
        can_sym_res = []
        can_p_res = []
        candidata_syms, candidata_p = symClasser.predict([sym_image]*3)
        for c_i in range(len(candidata_syms)):
            for c_j in range(len(candidata_syms[c_i])):
                if candidata_syms[c_i][c_j] not in can_sym_res:
                    can_sym_res.append(candidata_syms[c_i][c_j])
                    can_p_res.append(candidata_p[c_i][c_j])
        if len(can_sym_res)<1:
            print("ERROR: 有未识别出的字符")
        symsInfo = {}
        symsInfo["name"] = can_sym_res
        symsInfo["probability"] = can_p_res
        symsInfo["minRect"] = unknownSymbolsMinRects[i]
        symsInfo["symPixel"] = unknownSymbols[i]
        allSymbolsInfo.append(symsInfo)

    # 排好序
    allSymbolsInfo = sortSymbolsInfo(allSymbolsInfo)
    return allSymbolsInfo

# 给符号信息按照轴从左往右排
def sortSymbolsInfo(allSymbolsInfo):
    infos = allSymbolsInfo
    for i in range(len(infos)):
        for j in range(len(infos)-i-1):
            if infos[j]["minRect"][0]>infos[j+1]["minRect"][0]:
                tmp = infos[j]
                infos[j] = infos[j+1]
                infos[j+1] = tmp
    return infos

# 获取所有可能的数学公式函数
def getMEs(allSymbolsInfo,p):
    global points
    points = p
    if len(allSymbolsInfo)<1:
        return None
    mybeMEs = []
    if len(allSymbolsInfo)==1:
        for i in range(len(allSymbolsInfo[0]["name"])):
            meInfo = {}
            if allSymbolsInfo[0]["name"][i]=="line":
                meInfo["me"] = "-"
            else:
                meInfo["me"] = global_config.INFTYCDB_3_SHAPES_INFO[allSymbolsInfo[0]["name"][i]]["latex"]
            meInfo["probability"] = allSymbolsInfo[0]["probability"][i]
            mybeMEs.append(meInfo)
        return mybeMEs

    finals = []
    for symInfo in allSymbolsInfo:
        f = [0]*len(symInfo["name"])
        finals.append(f)
        print(symInfo["name"],symInfo["minRect"])
    print("finals:",finals)
    symbolsInfoMuti = []
    getSymbolsInfoSingle(symbolsInfoMuti, [], allSymbolsInfo, finals, 0)
    # for line in symbolsInfoMuti:
    #     for sym in line:
    #         print(sym["name"]+" ",end="")
    #     print()
    # print("symbolsInfoMuti_len:", len(symbolsInfoMuti))

    mes = []
    for meSymInfo in symbolsInfoMuti:
        # print("meSymInfo:", meSymInfo)
        print("合并函数前meSymInfo: ", end="")
        for db in meSymInfo:
            print(db["name"] + " ", end="")
        print()

        # 合并sin、lim等函数
        meSymInfo = mergeFunc(meSymInfo)

        print("合并函数后meSymInfo: ", end="")
        for db in meSymInfo:
            print(db["name"] + " ", end="")
        print()

        meTree = []
        mergeME(meSymInfo,meTree)
        print("meTree:",meTree)
        isLeg = judgeLegitimate(meTree)
        print("isLegitimate:",isLeg)
        meStr = translateMe(meTree)
        print("meStr:", meStr)
        if not isLeg or meStr in mes:
            continue
        mes.append(meStr)
    return mes

# 将数学公式的树形结构翻译为latex字符串形式
def translateMe(me):
    meStr = ""
    idx = 0
    for sym in me:
        filedsType = global_config.INFTYCDB_3_SHAPES_INFO[sym["name"]]["filedType"]
        if filedsType == "none":
            meStr += sym["latex"]+" "

        elif sym["name"]=="line":
            if len(sym["above"])>0 and len(sym["bottom"])>0:
                meStr+="\\frac{ "+translateMe(sym["above"])+"}{ "+translateMe(sym["bottom"])+"} "
            else:
                meStr+=sym["latex"]+" "
        elif sym["name"]=="sqrt":
            meStr+="\\sqrt{ "+translateMe(sym["content"])+"} "
            if len(sym["super"])>0:
                meStr+="^"+translateMe(sym["super"])+" "
        elif sym["name"]=="int":
            meStr += "\\int_{ "+translateMe(sym["rightBottom"])+"}^{ "+translateMe(sym["super"])+"} "
        elif sym["name"] == "lim":
            meStr += "\\lim_{ "+translateMe(sym["bottom"])+"} "
        elif sym["name"] == "Sigma":
            meStr += "\\sum_{ " + translateMe(sym["bottom"]) + "}^{ "+translateMe(sym["above"])+"} "
        elif sym["name"] == "LeftBrace":
            meStr += "\\left\{\\begin{array}{ll}"
            for brace_i in range(len(sym["right"])):
                meStr += "{ "+translateMe(sym["right"][brace_i]["left_formu"])+",} "
                if brace_i==len(sym["right"])-1:
                    meStr += "&{ "+translateMe(sym["right"][brace_i]["right_cond"])+"} "
                else:
                    meStr += "&{ " + translateMe(sym["right"][brace_i]["right_cond"]) + "}\\\ "
            meStr += "\\end{array}\\right"
        else:
            tmp = True
            if sym["name"] in global_config.NUMBER:
                pointMinRects = img_process.getMinRect(points)
                print("len(pointMinRects):",len(pointMinRects))
                for i in range(len(points)):
                    px, py, pw, ph = pointMinRects[i]
                    ox1, oy1, ow1, oh1 = sym["minRect"]
                    # if sym["name"]=="two":
                    #     print('sym["name"]:',sym["name"],"0<px-ox1-ow1<ow1",0<px-ox1-ow1<ow1,"idx<len(me)-1:",idx<len(me)-1,'me[idx+1]["name"]:',me[idx+1]["name"])
                    if 0<px-ox1-ow1<ow1 and py+ph-oy1-oh1<5 and py>oy1 and idx<len(me)-1 and (me[idx+1]["name"] in global_config.NUMBER):
                        # print('sym["name"]:',sym["name"])
                        ox2, oy2, ow2, oh2 = me[idx+1]["minRect"]
                        if 0<ox2-px-pw<ow2 and py+ph-oy2-oh2<5 and py>oy2:
                            meStr += sym["latex"] + " . "
                            tmp = False
            if tmp:
                meStr += sym["latex"]+" "
                filedsType_info = global_config.FILEDTYPE_INFO[filedsType]
                for filed in filedsType_info:
                    if len(sym[filed])>0:
                        if filed=="super":
                            meStr += "^ "+translateMe(sym["super"])
                        if filed=="sub":
                            meStr += "_{ " + translateMe(sym["sub"])+"} "
        # 索引值加一
        idx+=1
    return meStr


# 获取到所有可能的公式符号集
def getSymbolsInfoSingle(symbolsInfoMuti,symbolsInfoSingle,allSymbolsInfo,finals,idx):
    if len(allSymbolsInfo[idx]["name"])==0:
        tmp_symbolsInfoSingle = copy.deepcopy(symbolsInfoSingle)
        tmp_finals = copy.deepcopy(finals)
        symsInfo = {}
        symsInfo["name"] = "unknown"
        symsInfo["probability"] = "unknown"
        symsInfo["minRect"] = allSymbolsInfo[idx]["minRect"]
        symsInfo["symPixel"] = allSymbolsInfo[idx]["symPixel"]
        tmp_symbolsInfoSingle.append(symsInfo)
        if idx == len(allSymbolsInfo) - 1:
            symbolsInfoMuti.append(tmp_symbolsInfoSingle)
        else:
            getSymbolsInfoSingle(symbolsInfoMuti, tmp_symbolsInfoSingle, allSymbolsInfo, tmp_finals, idx + 1)
    else:
        for i in range(len(allSymbolsInfo[idx]["name"])):
            # print('allSymbolsInfo[idx]["name"]:',allSymbolsInfo[idx]["name"])
            # print("finals:",finals)
            if finals[idx][i]==0:
                tmp_symbolsInfoSingle = copy.deepcopy(symbolsInfoSingle)
                tmp_finals = copy.deepcopy(finals)
                symsInfo = {}
                symsInfo["name"] = allSymbolsInfo[idx]["name"][i]
                symsInfo["probability"] = allSymbolsInfo[idx]["probability"][i]
                symsInfo["minRect"] = allSymbolsInfo[idx]["minRect"]
                symsInfo["symPixel"] = allSymbolsInfo[idx]["symPixel"]
                tmp_symbolsInfoSingle.append(symsInfo)

                tmp_finals[idx][i] = 1
                if idx==len(allSymbolsInfo)-1:
                    symbolsInfoMuti.append(tmp_symbolsInfoSingle)
                else:
                    getSymbolsInfoSingle(symbolsInfoMuti, tmp_symbolsInfoSingle, allSymbolsInfo, tmp_finals, idx+1)

# 从ori中找出不在sub中的符号
def splitSyms(ori,sub):
    # 其他的符号
    otherInfo = []
    for info in ori:
        if info not in sub:
            otherInfo.append(copy.deepcopy(info))
    return otherInfo

# 合并如lim,cos,sin等函数
def mergeFunc(meSymInfo):
    Db = copy.deepcopy(meSymInfo)
    infos = []
    j = 0
    for i in range(len(Db)):
        if i<len(Db)-3 and j==0:
            # print("is lim:",(Db[i]["name"]=="l" or Db[i]["name"]=="one") and Db[i+1]["name"]=="i" and Db[i+2]["name"]=="m")
            # print("is lim:",not judgeIsMiddle(meSymInfo,Db[i],Db[i+1]))
            # print("is lim:",not judgeIsMiddle(meSymInfo,Db[i+1],Db[i+2]))
            # print((Db[i]["name"]=="l" or Db[i]["name"]=="one" or Db[i]["name"]=="i_like") and Db[i+1]["name"]=="i" and Db[i+2]["name"]=="m" and not judgeIsMiddle(meSymInfo,Db[i],Db[i+1]) and not judgeIsMiddle(meSymInfo,Db[i+1],Db[i+2]))
            if (Db[i]["name"]=="l" or Db[i]["name"]=="one" or Db[i]["name"]=="i_like") and Db[i+1]["name"]=="i" and Db[i+2]["name"]=="m" and not judgeIsMiddle2(meSymInfo,Db[i],Db[i+1]) and not judgeIsMiddle2(meSymInfo,Db[i+1],Db[i+2]):
                print("识别出lim")
                Db[i]["name"] = "lim"
                Db[i]["symPixel"] = Db[i]["symPixel"]+Db[i+1]["symPixel"]+Db[i+2]["symPixel"]
                # print(Db[i]["symPixel"])
                Db[i]["minRect"] = img_process.getMinRect([Db[i]["symPixel"]])[0]
                # infos.append(Db[i])
                j=3
            elif Db[i]["name"]=="c" and Db[i+1]["name"]=="o" and Db[i+2]["name"]=="s" and not judgeIsMiddle2(meSymInfo,Db[i],Db[i+1]) and not judgeIsMiddle2(meSymInfo,Db[i+1],Db[i+2]):
                print("识别出cos")
                Db[i]["name"] = "cos"
                Db[i]["symPixel"] = Db[i]["symPixel"]+Db[i+1]["symPixel"]+Db[i+2]["symPixel"]
                Db[i]["minRect"] = img_process.getMinRect([Db[i]["symPixel"]])[0]
                # infos.append(Db[i])
                j = 3
            elif Db[i]["name"]=="s" and Db[i+1]["name"]=="i" and Db[i+2]["name"]=="n" and not judgeIsMiddle2(meSymInfo,Db[i],Db[i+1]) and not judgeIsMiddle2(meSymInfo,Db[i+1],Db[i+2]):
                print("识别出sin")
                Db[i]["name"] = "sin"
                Db[i]["symPixel"] = Db[i]["symPixel"]+Db[i+1]["symPixel"]+Db[i+2]["symPixel"]
                Db[i]["minRect"] = img_process.getMinRect([Db[i]["symPixel"]])[0]
                # infos.append(Db[i])
                j = 3
        if j==0:
            infos.append(Db[i])
        elif j==3:
            infos.append(Db[i])
            j-=1
        else:
            j-=1

    return infos


# 将可能的符号合并为公式
def mergeME(meSymInfo,root):
    # 基线中的符号
    Db = constructDominantBaseline(meSymInfo,[])
    print("Db: ",end="")
    for db in Db:
        print(db["name"]+" ",end="")
    print()
    # 合并函数
    # Db = mergeFunc(Db, meSymInfo)
    # print("合并函数后Db: ", end="")
    # for db in Db:
    #     print(db["name"] + " ", end="")
    # print()
    # print("公式主线长度为:",len(Db))
    otherInfo = meSymInfo
    lastInfos = Db
    for info in Db:
        symInfo = {}
        symInfo["name"] = info["name"]
        symInfo["minRect"] = info["minRect"]
        # symInfo = copy.deepcopy(info)
        if symInfo["name"]=="line":
            symInfo["latex"] = "-"
        else:
            symInfo["latex"] = global_config.INFTYCDB_3_SHAPES_INFO[symInfo["name"]]["latex"]

        filedsType = global_config.INFTYCDB_3_SHAPES_INFO[info["name"]]["filedType"]
        if filedsType == "none":
            root.append(symInfo)
            continue
        filedsType_info = global_config.FILEDTYPE_INFO[filedsType]
        for filed in filedsType_info:
            symInfo[filed] = []
            thisInfos = []
            otherInfo = splitSyms(otherInfo, lastInfos)
            for sym in otherInfo:
                is_res = False
                if filed=="above":
                    is_res = isAbove(info,sym)
                elif filed=="bottom":
                    is_res = isBottom(info, sym)
                elif filed=="super":
                    is_res = isSuper(info, sym)
                elif filed=="sub":
                    is_res = isSub(info, sym)
                elif filed=="rightBottom":
                    is_res = isRightBottom(info, sym)
                elif filed=="content":
                    # print("content:",sym["name"])
                    is_res = isContent(info, sym)
                elif filed=="right":
                    is_res = isRight(info, sym)
                # elif filed=="leftBottom":
                #     is_res = isLeftBottom(info, sym)
                if is_res and not judgeIsMiddle(Db,info,sym):
                    thisInfos.append(copy.deepcopy(sym))
            if len(thisInfos)>0:
                print("thisInfos_len:",len(thisInfos))

                # 处理多项式
                if info["name"]=="LeftBrace":
                    x,y,w,h = info["minRect"]
                    right_img = glo_binary_img[:, x+ w+1:]
                    horProj = np.sum(right_img,axis=1)
                    # plt.plot(np.arange(len(horProj)),horProj)
                    # plt.show()
                    sp_ys = []
                    if horProj[0]>0:
                        sp_ys.append(0)
                    cu_blank = 0
                    for hor_i in range(1,len(horProj)):
                        if horProj[hor_i]>0 and horProj[hor_i-1]==0:
                            if hor_i-cu_blank<=3:
                                sp_ys.pop()
                            else:
                                sp_ys.append(hor_i)
                        elif horProj[hor_i]==0 and horProj[hor_i-1]>0:
                            sp_ys.append(hor_i-1)
                            cu_blank = hor_i
                    if horProj[len(horProj)-1]>0:
                        sp_ys.append(len(horProj)-1)
                    print("len(sp_ys):",len(sp_ys))
                    symInfo[filed] = [{"left_formu":[],"right_cond":[]} for _ in range(len(sp_ys)//2)]
                    for sp_ys_i in range(0,len(sp_ys),2):
                        hor_infos = []
                        for this_info in thisInfos:
                            x,y,w,h = this_info["minRect"]
                            # print('this_info["minRect"]:',this_info["minRect"])
                            if y+1>=sp_ys[sp_ys_i] and y+h-1<=sp_ys[sp_ys_i+1]:
                                hor_infos.append(this_info)
                        comma = 0
                        blank_max = 0
                        print("len(hor_infos):", len(hor_infos))
                        for hor_infos_i in range(len(hor_infos)-1):
                            if hor_infos[hor_infos_i+1]["minRect"][0]-hor_infos[hor_infos_i]["minRect"][0]>blank_max:
                                blank_max = hor_infos[hor_infos_i+1]["minRect"][0]-hor_infos[hor_infos_i]["minRect"][0]
                                comma = hor_infos_i
                        left_formu = []
                        right_cond = []
                        cx,cy,cw,ch = hor_infos[comma]["minRect"]
                        for i in range(len(hor_infos)):
                            hx,hy,hw,hh = hor_infos[i]["minRect"]
                            if hx+hw<cx:
                                left_formu.append(hor_infos[i])
                            elif hx>cx+cw:
                                right_cond.append(hor_infos[i])

                        mergeME(left_formu, symInfo[filed][sp_ys_i//2]["left_formu"])
                        mergeME(right_cond, symInfo[filed][sp_ys_i // 2]["right_cond"])
                else:
                    mergeME(thisInfos, symInfo[filed])
            lastInfos = thisInfos

        root.append(symInfo)

# 判断sym和info间是间隔着Db中的字符,用于判断是否为上标或下标
def judgeIsMiddle(Db,info,sym):
    for d in Db:
        if d["minRect"][0]>=info["minRect"][0]+info["minRect"][2] and d["minRect"][0]<=sym["minRect"][0]:
            return True
    return False

# 判断sym和info间是间隔着Db中的字符,用于判断是否为上标或下标
def judgeIsMiddle2(Db,info,sym):
    for d in Db:
        if d["minRect"][0]>=info["minRect"][0]+info["minRect"][2] and d["minRect"][0]+d["minRect"][2]<=sym["minRect"][0]:
            return True
    return False

# 构造基线函数
def constructDominantBaseline(meSymInfo,Db):
    if len(Db)==0:
        ldb = getDominantSymbol(meSymInfo)
        Db.append(ldb)
    # print("Db:",Db)
    s = Db[-1]
    if s["name"]=="LeftBrace":
        return Db
    HS = getRightNeighbors(s,meSymInfo)
    if len(HS)==0:
        return Db
    sd = getDominantSymbol(HS)
    # print("HS:", HS,"sd:",sd)
    Db.append(sd)
    return constructDominantBaseline(meSymInfo, Db)

# 从meSymInfo中获取s的右邻居
def getRightNeighbors(s,meSymInfo):
    HS = []
    for sym in meSymInfo:
        if sym["minRect"][0]>s["minRect"][0]+s["minRect"][2]:
            if isRightHor(s, sym):
                HS.append(sym)
    return HS

# 获取基线最左边的元素
def getDominantSymbol(meSymInfo):
    info = copy.deepcopy(meSymInfo)
    info_len = len(info)
    if info_len==1:
        return info[0]
    if isDominant(meSymInfo[1],meSymInfo[0]):
        info = info[1:]
    else:
        info = info[:1]+info[2:]
    return getDominantSymbol(info)

# 判断obj字符是否被ori字符所控制
def isDominant(ori,obj):
    if ori["name"]=="line" and obj["name"]=="line" and ori["minRect"][2]<obj["minRect"][2]:
        return False
    res = False
    filedsType = global_config.INFTYCDB_3_SHAPES_INFO[ori["name"]]["filedType"]
    if filedsType=="none":
        return res
    filedsType_info = global_config.FILEDTYPE_INFO[filedsType]
    for filed in filedsType_info:
        if filed=="above":
            res = res or isAbove(ori,obj)
        elif filed=="bottom":
            res = res or isBottom(ori, obj)
        elif filed=="sub":
            res = res or isSub(ori, obj)
        elif filed=="super":
            res = res or isSuper(ori, obj)
        elif filed=="content":
            res = res or isContent(ori, obj)
        elif filed=="rightBottom":
            res = res or isRightBottom(ori,obj)
        elif filed=="right":
            res = res or isRight(ori,obj)
    return res

# 用于多项式大括号
def isRight(ori,obj):
    return ori["minRect"][0]+ori["minRect"][2]<obj["minRect"][0]

# 用于判断积分中的下标
def isRightBottom(ori,obj):
    # print((ori["minRect"][0]+ori["minRect"][2])/2,obj["minRect"][0])
    # print("isRightBottom("+ori["name"]+","+obj["name"]+"):",(ori["minRect"][0]+ori["minRect"][2])/2<obj["minRect"][0] and (ori["minRect"][1]+ori["minRect"][3])/2<obj["minRect"][1] and ori["minRect"][1]+ori["minRect"][3]>obj["minRect"][1] and ori["minRect"][0]+ori["minRect"][2]>obj["minRect"][0])
    return (ori["minRect"][0]+ori["minRect"][2]/2)<obj["minRect"][0] and (ori["minRect"][1]+ori["minRect"][3]/2)<obj["minRect"][1] and ori["minRect"][1]+ori["minRect"][3]>obj["minRect"][1] and ori["minRect"][0]+ori["minRect"][2]>obj["minRect"][0]

# 判断obj是否属于ori的上方关系
def isAbove(ori,obj):
    return ori["minRect"][1]>obj["minRect"][1]+obj["minRect"][3] and np.abs(ori["minRect"][0]-obj["minRect"][0])<3*np.max([ori["minRect"][2],obj["minRect"][2]])
    # print("isAbove("+ori["name"]+","+obj["name"]+"):",ori["minRect"][0]<obj["minRect"][0] and ori["minRect"][0]+ori["minRect"][2]>obj["minRect"][0]+obj["minRect"][2] and ori["minRect"][1]>obj["minRect"][1]+obj["minRect"][3])
    # return ori["minRect"][0]<obj["minRect"][0] and ori["minRect"][0]+ori["minRect"][2]>obj["minRect"][0]+obj["minRect"][2] and ori["minRect"][1]>obj["minRect"][1]+obj["minRect"][3]

# 判断obj是否和ori为平行关系，且在ori的右侧
def isRightHor(ori,obj):
    shapes_infos = global_config.INFTYCDB_3_SHAPES_INFO
    types_info = global_config.TYPE_INFO
    ori_type = shapes_infos[ori["name"]]["type"]
    ori_type_sup = types_info[ori_type]["superThreshold"]
    ori_type_sub = types_info[ori_type]["subThreshold"]
    ori_type_cen = types_info[ori_type]["centroid"]
    ori_sup = ori["minRect"][1]+ori_type_sup[3]*ori["minRect"][3]
    ori_sub = ori["minRect"][1]+ori_type_sub[3]*ori["minRect"][3]
    ori_centerY = ori["minRect"][1] + ori_type_cen[3] * ori["minRect"][3]

    obj_type = shapes_infos[obj["name"]]["type"]
    obj_type_cen = types_info[obj_type]["centroid"]
    obj_centerY = obj["minRect"][1] + obj_type_cen[3] * obj["minRect"][3]
    # 打印输出，查看效果
    # print(ori["name"]+"_centerY:",ori_centerY,obj["name"]+"_centerY:",obj_centerY,ori["name"])
    # print("isRightHor("+ori["name"]+","+obj["name"]+"):",obj_centerY<ori_sub and obj_centerY>ori_sup and obj["minRect"][0]>(ori["minRect"][0]+ori["minRect"][2]) and (np.abs(ori_centerY-obj_centerY)<=((ori_centerY-ori["minRect"][1])/10) or (ori["minRect"][1]<ori_sup and ori["minRect"][1]+ori["minRect"][3]>ori_sub) or (ori["minRect"][1]>ori_sup and ori["minRect"][1]+ori["minRect"][3]<ori_sub)))
    return (obj_centerY<ori_sub and obj_centerY>ori_sup and obj["minRect"][0]>(ori["minRect"][0]+ori["minRect"][2]) and (np.abs(ori_centerY-obj_centerY)<=((ori_centerY-ori["minRect"][1])/5)) or (obj["minRect"][1]<ori_sup and obj["minRect"][1]+obj["minRect"][3]>ori_sub) or (obj["minRect"][1]>ori_sup and obj["minRect"][1]+obj["minRect"][3]<ori_sub))

# 判断是否为上标
def isSuper(ori,obj):
    shapes_infos = global_config.INFTYCDB_3_SHAPES_INFO
    types_info = global_config.TYPE_INFO
    ori_type = shapes_infos[ori["name"]]["type"]
    ori_type_sup = types_info[ori_type]["superThreshold"]
    ori_sup = ori["minRect"][1] + ori_type_sup[3] * ori["minRect"][3]

    obj_type = shapes_infos[obj["name"]]["type"]
    obj_type_cen = types_info[obj_type]["centroid"]
    obj_centerY = obj["minRect"][1] + obj_type_cen[3] * obj["minRect"][3]
    # print("isSuper("+ori["name"]+","+obj["name"]+"):",obj["minRect"][0]>(ori["minRect"][0]+ori["minRect"][2]) and obj_centerY<ori_sup)
    return obj["minRect"][0]>(ori["minRect"][0]+ori["minRect"][2]) and obj_centerY<ori_sup

# 判断是否为下标
def isSub(ori,obj):
    shapes_infos = global_config.INFTYCDB_3_SHAPES_INFO
    types_info = global_config.TYPE_INFO
    ori_type = shapes_infos[ori["name"]]["type"]
    ori_type_sub = types_info[ori_type]["subThreshold"]
    ori_sub = ori["minRect"][1] + ori_type_sub[3] * ori["minRect"][3]

    obj_type = shapes_infos[obj["name"]]["type"]
    obj_type_cen = types_info[obj_type]["centroid"]
    obj_centerY = obj["minRect"][1] + obj_type_cen[3] * obj["minRect"][3]

    # print("isSub("+ori["name"]+","+obj["name"]+"):",obj["minRect"][0]>(ori["minRect"][0]+ori["minRect"][2]) and obj_centerY>ori_sub)
    return obj["minRect"][0]>(ori["minRect"][0]+ori["minRect"][2]) and obj_centerY>ori_sub

# 判断obj是否被ori包含
def isContent(ori,obj):
    shapes_infos = global_config.INFTYCDB_3_SHAPES_INFO
    types_info = global_config.TYPE_INFO
    obj_type = shapes_infos[obj["name"]]["type"]
    obj_type_cen = types_info[obj_type]["centroid"]
    obj_centerY = obj["minRect"][1] + obj_type_cen[3] * obj["minRect"][3]
    obj_centerX = obj["minRect"][0] + obj_type_cen[2] * obj["minRect"][2]
    return ori["minRect"][0]<obj_centerX and ori["minRect"][1]<obj_centerY and ori["minRect"][1]+ori["minRect"][3]>obj_centerY and ori["minRect"][0]+ori["minRect"][2]>obj_centerX
    # print("isContent(" + ori["name"] + "," + obj["name"] + "):",ori["minRect"][0]<obj["minRect"][0] and ori["minRect"][1]<obj["minRect"][1] and ori["minRect"][1]+ori["minRect"][3]>obj["minRect"][1]+obj["minRect"][3] and ori["minRect"][0]+ori["minRect"][2]>obj["minRect"][0]+obj["minRect"][2])
    # return ori["minRect"][0]<obj["minRect"][0] and ori["minRect"][1]<obj["minRect"][1] and ori["minRect"][1]+ori["minRect"][3]>obj["minRect"][1]+obj["minRect"][3] and ori["minRect"][0]+ori["minRect"][2]>obj["minRect"][0]+obj["minRect"][2]

# 判断是否为下方关系
def isBottom(ori,obj):
    return ori["minRect"][1]+ori["minRect"][3]<obj["minRect"][1] and np.abs(ori["minRect"][0]-obj["minRect"][0])<3*np.max([ori["minRect"][2],obj["minRect"][2]])
    # print("isBottom("+ori["name"]+","+obj["name"]+"):",ori["minRect"][0]<obj["minRect"][0] and ori["minRect"][0]+ori["minRect"][2]>obj["minRect"][0]+obj["minRect"][2] and ori["minRect"][1]+ori["minRect"][3]<obj["minRect"][1])
    # return ori["minRect"][0]<obj["minRect"][0] and ori["minRect"][0]+ori["minRect"][2]>obj["minRect"][0]+obj["minRect"][2] and ori["minRect"][1]+ori["minRect"][3]<obj["minRect"][1]

# 检测数学公式的合法性
def judgeLegitimate(meTree):
    res = True
    # 运算符栈
    operator = []
    # 操作数栈
    operand = []
    # 括号栈
    par = []
    # 绝对值括号
    verline = []
    idx = 0
    for sym in meTree:
        if sym["name"]=="line":
            if len(sym["above"])>0 and len(sym["bottom"])>0:
                res = res and judgeLegitimate(sym["above"]) and judgeLegitimate(sym["bottom"])
            elif (len(sym["above"])>0 and len(sym["bottom"])==0) or (len(sym["above"])==0 and len(sym["bottom"])>0):
                return False
            else:
                # 操作符前必须有操作数
                if idx<len(meTree)-1 and meTree[idx+1]["name"] in global_config.OPERAND:
                    idx+=1
                    continue
                if len(operand)==0:
                    return False
                if len(operator)==0:
                    operator.append(sym)
                else:
                    return False
        elif sym["name"]=="sqrt":
            if len(sym["content"])==0:
                return False
            else:
                if len(sym["super"]) > 0:
                    res = res and judgeLegitimate(sym["super"]) and judgeLegitimate(sym["content"])
                else:
                    res = res and judgeLegitimate(sym["content"])
        elif sym["name"]=="int":
            if (len(sym["rightBottom"])>0 and len(sym["super"])==0) or (len(sym["rightBottom"])==0 and len(sym["super"])>0) or (len(sym["rightBottom"])==0 and len(sym["super"])==0):
                return False
            else:
                res = res and judgeLegitimate(sym["rightBottom"]) and judgeLegitimate(sym["super"])
        elif sym["name"] == "lim":
            if len(sym["bottom"])==0:
                return False
            else:
                res = res and judgeLegitimate(sym["bottom"])
        elif sym["name"] == "Sigma":
            if (len(sym["above"])>0 and len(sym["bottom"])==0) or (len(sym["above"])==0 and len(sym["bottom"])>0) or (len(sym["bottom"])==0 and len(sym["above"])==0):
                return False
            else:
                res = res and judgeLegitimate(sym["above"]) and judgeLegitimate(sym["bottom"])
        elif sym["name"] == "cos" or sym["name"] == "sin":
            if idx>=len(meTree)-2:
                return False
            if meTree[idx+1]["name"]!="LeftPar":
                return False
        elif sym["name"]=="LeftPar":
            par.append(sym)
        elif sym["name"]=="RightPar":
            if len(par)!=1:
                return False
            if par[0]["name"]!="LeftPar":
                return False
            else:
                par.pop()
        elif sym["name"]=="verLine":
            if len(verline)==0:
                verline.append(sym)
            elif len(verline)==1:
                verline.pop()
            else:
                return False
        elif sym["name"] == "LeftBrace":
            if len(sym["right"])==0:
                return False
            for brace_i in range(len(sym["right"])):
                res = res and judgeLegitimate(sym["right"][brace_i]["left_formu"])
                res = res and judgeLegitimate(sym["right"][brace_i]["right_cond"])

        # 是操作数
        elif sym["name"] in global_config.OPERAND:
            if len(operand)==0:
                operand.append(sym)
            else:
                if len(operator)==0 and len(par)==0 and len(verline)==0:
                    operand.append(sym)
                else:
                    operand.clear()
                    operand.append(sym)
                    if len(operator)==1:
                        operator.pop()
            if "super" in sym and len(sym["super"])>0:
                res = res and judgeLegitimate(sym["super"])
            if "sub" in sym and len(sym["sub"])>0:
                res = res and judgeLegitimate(sym["sub"])
        # 是操作符
        elif sym["name"] in global_config.OPERATOR:
            # 操作符前必须有操作数
            if sym["name"]=="plus" and idx<len(meTree)-1 and (meTree[idx+1]["name"] in global_config.OPERAND):
                idx += 1
                continue
            if len(operand) == 0:
                return False
            if len(operator) == 0:
                operator.append(sym)
            else:
                return False
        # 索引值加一
        idx+=1
    if len(operator)>0 or len(verline)>0 or len(par)>0:
        return False
    return res
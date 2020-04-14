import cv2
import copy
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
import img_process
import global_config
from global_variable import symClasser

# 初步锁定可能的杂质区域
# 默认图片中公式所占的比例最大，且杂质距离公式有一定距离
def maybeImpurity(image):
    # 将所有轮廓的宽度改为1个像素
    one_weight = skeletonize(image)  # skeletonize函数用于将图像轮廓转为宽度为1个像素的图像
    # 分割得到的符号集合
    symbols = []
    # 水平投影
    horizontal_projection = np.sum(one_weight, axis=1)
    # 垂直投影
    vertical_projection = np.sum(one_weight, axis=0)
    expression_ver = vertical_projection[vertical_projection < 2]
    # plt.imshow(one_weight,cmap="Greys", interpolation="None")
    plt.plot(np.arange(one_weight.shape[1]),vertical_projection)
    # plt.scatter(np.arange(binary_img.shape[1])[vertical_projection<2],expression_ver)
    plt.show()
    return image

# 使用8连通域进行图像分割
def getSymbols(image):
    symbols = []
    final = np.zeros(image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if final[x][y]==0 and image[x][y]==1:
                symbol = []
                stack = [[x,y]]
                while stack:
                    x,y = stack.pop()
                    symbol.append([x,y])
                    final[x][y] = 1
                    dfSearch(image,final,stack,x,y)
                # 返回先按照x再按照y排好序的字符坐标数组
                symbol = sorted(symbol)
                symbols.append(symbol)

    return sorted(symbols)

# 广度遍历
def dfSearch(image,final,stack,x,y):
    if x>0 and x<image.shape[0]-1:
        if final[x + 1][y] == 0 and image[x + 1][y] == 1:
            stack.append([x+1,y])
            final[x + 1][y] = 1
        if final[x - 1][y] == 0 and image[x - 1][y] == 1:
            stack.append([x-1,y])
            final[x - 1][y]=1
        if y>0 and y<image.shape[1]-1:
            if final[x][y+1]==0 and image[x][y+1]==1:
                stack.append([x, y+1])
                final[x][y+1] = 1
            if final[x+1][y-1]==0 and image[x+1][y-1]==1:
                stack.append([x+1, y-1])
                final[x+1][y-1] = 1
            if final[x+1][y+1]==0 and image[x+1][y+1]==1:
                stack.append([x+1, y + 1])
                final[x+1][y+1] = 1
            if final[x][y-1]==0 and image[x][y-1]==1:
                stack.append([x, y-1])
                final[x][y-1] = 1
            if final[x-1][y-1]==0 and image[x-1][y-1]==1:
                stack.append([x-1, y-1])
                final[x-1][y-1] = 1
            if final[x-1][y+1]==0 and image[x-1][y+1]==1:
                stack.append([x-1, y+1])
                final[x-1][y+1] = 1
        elif y==0:
            if final[x][y+1]==0 and image[x][y+1]==1:
                stack.append([x,y+1])
                final[x][y+1] = 1
            if final[x+1][y+1]==0 and image[x+1][y+1]==1:
                stack.append([x+1, y+1])
                final[x+1][y+1] = 1
            if final[x-1][y+1]==0 and image[x-1][y+1]==1:
                stack.append([x-1, y+1])
                final[x-1][y+1] = 1
        elif y==image.shape[1]-1:
            if final[x+1][y-1]==0 and image[x+1][y-1]==1:
                stack.append([x+1, y-1])
                final[x+1][y-1] = 1
            if final[x][y-1]==0 and image[x][y-1]==1:
                stack.append([x, y-1])
                final[x][y-1] = 1
            if final[x-1][y-1]==0 and image[x-1][y-1]==1:
                stack.append([x-1, y-1])
                final[x-1][y-1] = 1
    elif x==0:
        if final[x + 1][y] == 0 and image[x + 1][y] == 1:
            stack.append([x + 1, y])
            final[x+1][y] = 1
        if y>0 and y<image.shape[1]-1:
            if final[x][y+1]==0 and image[x][y+1]==1:
                stack.append([x,y+1])
                final[x][y+1] = 1
            if final[x+1][y-1]==0 and image[x+1][y-1]==1:
                stack.append([x+1, y-1])
                final[x+1][y-1] = 1
            if final[x+1][y+1]==0 and image[x+1][y+1]==1:
                stack.append([x+1, y+1])
                final[x+1][y+1] = 1
            if final[x][y-1]==0 and image[x][y-1]==1:
                stack.append([x, y-1])
                final[x][y-1] = 1
        elif y==0:
            if final[x][y+1]==0 and image[x][y+1]==1:
                stack.append([x,y+1])
                final[x][y+1] = 1
            if final[x+1][y+1]==0 and image[x+1][y+1]==1:
                stack.append([x+1, y+1])
                final[x+1][y+1] = 1
        elif y==image.shape[1]-1:
            if final[x+1][y-1]==0 and image[x+1][y-1]==1:
                stack.append([x+1, y-1])
                final[x+1][y-1] = 1
            if final[x][y-1]==0 and image[x][y-1]==1:
                stack.append([x, y-1])
                final[x][y-1] = 1
    elif x==image.shape[0]-1:
        if final[x - 1][y] == 0 and image[x - 1][y] == 1:
            stack.append([x - 1, y])
            final[x-1][y] = 1
        if y>0 and y<image.shape[1]-1:
            if final[x][y+1]==0 and image[x][y+1]==1:
                stack.append([x,y+1])
                final[x][y+1] = 1
            if final[x][y-1]==0 and image[x][y-1]==1:
                stack.append([x, y-1])
                final[x][y-1] = 1
            if final[x-1][y-1]==0 and image[x-1][y-1]==1:
                stack.append([x-1, y-1])
                final[x-1][y-1] = 1
            if final[x-1][y+1]==0 and image[x-1][y+1]==1:
                stack.append([x-1, y+1])
                final[x-1][y+1] = 1
        elif y==0:
            if final[x][y+1]==0 and image[x][y+1]==1:
                stack.append([x,y+1])
                final[x][y+1] = 1
            if final[x-1][y+1]==0 and image[x-1][y+1]==1:
                stack.append([x-1, y+1])
                final[x-1][y+1] = 1
        elif y==image.shape[1]-1:
            if final[x][y-1]==0 and image[x][y-1]==1:
                stack.append([x, y-1])
                final[x][y-1] = 1
            if final[x-1][y-1]==0 and image[x-1][y-1]==1:
                stack.append([x-1, y-1])
                final[x-1][y-1] = 1

# 绘制符号的最小矩形框
def drawSymbolsRect(minRects,image):
    ori_image = image
    for x,y,w,h in minRects:
        cv2.rectangle(ori_image, (x, y), (x + w,y + h), (0, 255, 0), 1)
    plt.imshow(ori_image, cmap="Greys", interpolation="None")
    plt.show()

# 判断符号是否为根号
# image为二值图
def isSqrt(minRect,image):
    x,y,w,h = minRect
    symbol_img = image[y:y + h, x:x + w]

# 识别横线
def recogLine(img,rect):
    x,y,w,h = rect
    # print(x,len(img),w,h,len(img)/(w*h),w/h)
    if (len(img)/(w*h)>=0.85 or h<=3) and w/h>=4:
        return True
    return False

# 识别竖线
def recogVerLine(img,rect):
    x,y,w,h = rect
    # print(x,len(img),w,h,len(img)/(w*h),w/h)
    if (len(img)/(w*h)>=0.9 or w<=3) and h/w>=4:
        return True
    return False

# 识别圆点
def recogPoint(img,rect):
    x,y,w,h = rect
    if ((len(img)/(w*h)>=0.80 and ([y+h//2,x+w//2] in img) and w/h>=4/5 and w/h<=5/4) or (w<=4 and h<=4)):
        return True
    return False

# 合并等于符号
def mergeEqual(symbols):
    minRects = img_process.getMinRect(symbols)
    if len(symbols)<=1:
        return symbols,[],[]
    mybeLineSym = []
    mybeLineRec = []
    notLineSym = []
    for i in range(len(symbols)):
        symbol = symbols[i]
        minRect = minRects[i]
        if recogLine(symbol,minRect):
            mybeLineSym.append(symbols[i])
            mybeLineRec.append(minRects[i])
        else:
            notLineSym.append(symbols[i])

    print("mybeLineLen:",len(mybeLineSym))
    if len(mybeLineSym)<=1:
        return symbols,[],[]

    final = np.zeros(len(mybeLineSym))
    for i in range(len(mybeLineSym)-1):
        for j in range(i+1,len(mybeLineSym)):
            x1,y1,w1,h1 = mybeLineRec[i]
            x2,y2,w2,h2 = mybeLineRec[j]
            if final[i]<2 and final[j]<2 and np.abs(x1-x2)<=2 and np.abs(w1-w2)<=2:
                mybeLineSym[i] = np.append(np.array(mybeLineSym[i]),np.array(mybeLineSym[j]),axis=0).tolist()
                mybeLineRec[i] = img_process.getMinRect([mybeLineSym[i]])[0]
                if final[i]==1:
                    final[i] = 2
                elif final[i]==0:
                    final[i] = 1
                final[j] = 3
    equals = [] #等于符号
    equivs = [] #恒等于符号
    for i in range(len(mybeLineSym)):
        if final[i]==0:
            notLineSym.append(mybeLineSym[i])
        elif final[i]==1:
            equals.append(mybeLineSym[i])
        elif final[i]==2:
            equivs.append(mybeLineSym[i])
    return notLineSym,equals,equivs

# 合并除号函数、点、横线、竖线
def mergeDiv(symbols):
    minRects = img_process.getMinRect(symbols)
    if len(minRects)<=0:
        return symbols,[],[],[],[]
    lines = []
    lineMinRects = []
    points = []
    pointMinRects = []
    normalSyms = []
    verLines = []
    for i in range(len(symbols)):
        if recogLine(symbols[i],minRects[i]):
            lines.append(symbols[i])
            lineMinRects.append(minRects[i])
        elif recogPoint(symbols[i],minRects[i]):
            points.append(symbols[i])
            pointMinRects.append(minRects[i])
        elif recogVerLine(symbols[i],minRects[i]):
            verLines.append(symbols[i])
        else:
            normalSyms.append(symbols[i])

    print("mybeLineLen:",len(lines))
    print("mybePoint_len:", len(points))
    print("normalSyms_len:",len(normalSyms))
    if len(lines)<1 or len(points)<=1:
        return normalSyms,[],lines,points,verLines

    # 合并除号
    pointFinal = np.zeros(len(points))
    lineFinal = np.zeros(len(lines))
    for i in range(len(lines)):
        for j in range(len(points)-1):
            lx,ly,lw,lh = lineMinRects[i]
            px,py,pw,ph = pointMinRects[j]
            if pointFinal[j]==0 and px>lx and px+pw<lx+lw and pw/lw<0.35 and (ly+lh<py or py+ph<ly) and np.abs(px-lx-lw/2)<5 and lw/pw<20:
                for k in range(j+1,len(points)):
                    p2x,p2y,p2w,p2h = pointMinRects[k]
                    if pointFinal[k]==0 and p2x > lx and p2x + p2w < lx + lw  and p2w / lw < 0.35 and (ly+lh<p2y or p2y+p2h<ly) and np.abs(p2x-lx-lw/2)<5 and lw/p2w<20:
                        lines[i] = np.append(np.array(lines[i]),np.array(points[j]),axis=0).tolist()
                        lines[i] = np.append(np.array(lines[i]),np.array(points[k]),axis=0).tolist()
                        lineFinal[i] = 1
                        pointFinal[j]=1
                        pointFinal[k]=1

    divs = []
    re_lines = []
    re_points = []
    for i in range(len(lines)):
        if lineFinal[i]==0:
            re_lines.append(lines[i])
        else:
            divs.append(lines[i])

    for i in range(len(points)):
        if pointFinal[i]==0:
            re_points.append(points[i])
    return normalSyms,divs,re_lines,re_points,verLines

# 合并因像素不连续断裂的字符，明显相互包含的字符，且不为根号
def mergeSplitSymbol(par_symbols,binary_img):
    symbols = par_symbols
    minRects = img_process.getMinRect(symbols)
    if len(minRects)<=1:
        return symbols
    final = np.zeros(len(symbols))
    isSqrt = [-1]*len(symbols)
    isInt = [-1]*len(symbols)
    for i in range(len(minRects)-1):
        for j in range(i+1,len(minRects)):
            xi,yi,wi,hi = minRects[i]
            xj,yj,wj,hj = minRects[j]
            # 当符号有包含关系时，使用训练好的模型预测较大的符号是否为根号
            if final[j]==0 and final[i]==0 and xi<=xj and yi<=yj and xi+wi>=xj+wj and yi+hi>=yj+hj and isSqrt[i]!=1 and isInt[i]!=1:
                if isSqrt[i]==-1 and isInt[i]==-1:
                    cand_res = []
                    sym_image = img_process.getSymbolImg(symbols[i], minRects[i], global_config.IMG_SIZE, binary_img)
                    # plt.imshow(sym_image, cmap="Greys", interpolation="None")
                    # plt.show()
                    sym_image = sym_image.reshape((sym_image.shape[0] * sym_image.shape[1],))
                    for _ in range(6):
                        candidata_syms,candidata_p = symClasser.predict([sym_image])
                        cand_res+=candidata_syms[0]
                    print("cand_res:", cand_res)
                    # 因为分类器认为pi和sqrt很像，且有包含关系的字符不可能为pi
                    if "sqrt" or "pi" in cand_res:
                        isSqrt[i] = 1
                        break
                    elif "int" in cand_res:
                        isInt[i] = 1
                        break
                isSqrt[i] = 0
                isInt[i] = 0
                final[j] = 1
                symbols[i] = symbols[i] + symbols[j]
            elif final[j]==0 and final[i]==0 and xi>xj and yi>yj and xi+wi<xj+wj and yi+hi<yj+hj and isSqrt[j]!=1 and isInt[j]!=1:
                if isSqrt[j]==-1 and isInt[j]==-1:
                    cand_res = []
                    sym_image = img_process.getSymbolImg(symbols[j], minRects[j], global_config.IMG_SIZE, binary_img)
                    sym_image = sym_image.reshape((sym_image.shape[0] * sym_image.shape[1],))
                    for _ in range(6):
                        candidata_syms,candidata_p = symClasser.predict([sym_image])
                        cand_res+=candidata_syms[0]
                    print("cand_res:", cand_res)
                    if "sqrt" or "pi" in cand_res:
                        isSqrt[j] = 1
                        print("candidata_syms[0]:",candidata_syms[0])
                        continue
                    elif "int" in cand_res:
                        isInt[j] = 1
                        print("candidata_syms[0]:", candidata_syms[0])
                        continue
                isSqrt[j] = 0
                isInt[j] = 0
                final[i] = 1
                symbols[j] = symbols[i] + symbols[j]
                break
            # 重叠情况时，根据两个分割符号的大小以及重叠大小进行合并
            elif final[j]==0 and final[i]==0 and not (xi+wi<xj or xj+wj<xi or yi+hi<yj or yj+hj<yi) and isSqrt[i]!=1 and isInt[i]!=1 and isSqrt[j]!=1 and isInt[j]!=1:
                print("wj/hj:",wj/hj)

                if wi>wj and hi>hj and wj/hj<=2 and wj/hj>=0.5:
                    if isSqrt[i]==-1 and isInt[i]==-1:
                        cand_res = []
                        sym_imagei = img_process.getSymbolImg(symbols[i], minRects[i], global_config.IMG_SIZE,
                                                              binary_img)
                        sym_imagei = sym_imagei.reshape((sym_imagei.shape[0] * sym_imagei.shape[1],))
                        for _ in range(6):
                            candidata_syms, candidata_p = symClasser.predict([sym_imagei])
                            cand_res+=candidata_syms[0]
                        if "sqrt" in cand_res:
                            isSqrt[i] = 1
                            break
                        elif "int" in cand_res:
                            isInt[i] = 1
                            break
                    isSqrt[i] = 0
                    isInt[i] = 0
                    symbols[i] = symbols[i] + symbols[j]
                    final[j] = 1
                elif wi<wj and hi<hj and wi/hi<=2 and wi/hi>=0.5:
                    if isSqrt[j]==-1 and isInt[j]==-1:
                        cand_res = []
                        sym_jmagej = img_process.getSymbolImg(symbols[j], minRects[j], global_config.IMG_SIZE,
                                                              binary_img)
                        sym_jmagej = sym_jmagej.reshape((sym_jmagej.shape[0] * sym_jmagej.shape[1],))
                        for _ in range(6):
                            candidata_syms, candidata_p = symClasser.predict([sym_jmagej])
                            cand_res += candidata_syms[0]
                        if "sqrt" in cand_res:
                            isSqrt[j] = 1
                            continue
                        elif "int" in cand_res:
                            isInt[j] = 1
                            continue
                    isSqrt[j] = 0
                    isInt[j] = 0
                    symbols[j] = symbols[i] + symbols[j]
                    final[i] = 1
                    break
            elif final[j]==0 and final[i]==0:
                if yi>yj and 0<yi-yj-hj<np.min((hi,hj,10)) and (not (xi+wi+1<xj or xj+wj+1<xi)):
                    symbols[i] = symbols[i] + symbols[j]
                    final[j] = 1
                elif yi<yj and 0<yj-yi-hi<np.min((hi,hj,10)) and (not (xi+wi+1<xj or xj+wj+1<xi)):
                    symbols[i] = symbols[i] + symbols[j]
                    final[j] = 1
    res = []
    mybe_sqrts = []
    mybe_ints = []
    for i in range(len(symbols)):
        if final[i]==0:
            if isSqrt[i]==1:
                mybe_sqrts.append(symbols[i])
            elif isInt[i]==1:
                mybe_ints.append(symbols[i])
            else:
                res.append(symbols[i])

    return res,mybe_sqrts,mybe_ints

# 合并i和j符号
def mergeiandj(unKnownSymbols,points,binary_img):
    final_p = np.zeros(len(points))
    final_sym = np.zeros(len(unKnownSymbols))
    iSyms = []
    jSyms = []
    for i in range(len(points)):
        p = points[i]
        for j in range(len(unKnownSymbols)):
            if final_sym[j]==1:
                continue
            sym = unKnownSymbols[j]
            xp, yp, wp, hp = img_process.getMinRect([p])[0]
            xsym, ysym, wsym, hsym = img_process.getMinRect([sym])[0]
            if xsym<xp and xp-(xsym+wsym)<wsym/2 and yp+hp<ysym and hsym/wsym>=1.2 and (ysym-yp-hp)<hsym/2:
                sym_image = img_process.getSymbolImg(sym,(xsym, ysym, wsym, hsym), global_config.IMG_SIZE, binary_img)
                # plt.imshow(sym_image, cmap="Greys", interpolation="None")
                # plt.show()
                sym_image = sym_image.reshape((sym_image.shape[0] * sym_image.shape[1],))
                candidata_syms, candidata_p = symClasser.predict([sym_image]*3)
                can_sym_res = []
                can_p_res = []
                for c_i in range(len(candidata_syms)):
                    for c_j in range(len(candidata_syms[c_i])):
                        if candidata_syms[c_i][c_j] not in can_sym_res:
                            can_sym_res.append(candidata_syms[c_i][c_j])
                            can_p_res.append(candidata_p[c_i][c_j])

                print(can_sym_res, can_p_res)
                if "l" in can_sym_res or "one" in can_sym_res or "t" in can_sym_res:
                    tmpi = copy.deepcopy(p+sym)
                    iSyms.append(tmpi)
                    final_sym[j] = 1
                    final_p[i] = 1
                    break
                if "j" in can_sym_res:
                    tmpj = copy.deepcopy(p + sym)
                    jSyms.append(tmpj)
                    final_sym[j] = 1
                    final_p[i] = 1
                    break
    tmp_unKnownSymbols = []
    for i in range(len(unKnownSymbols)):
        if final_sym[i]==0:
            tmp_unKnownSymbols.append(unKnownSymbols[i])
    tmp_points = []
    for i in range(len(points)):
        if final_p[i]==0:
            tmp_points.append(points[i])
    return tmp_unKnownSymbols,tmp_points,iSyms,jSyms

# 合并大于等于和小于等于符号
def mergeLeg_Gep(unKnownSymbols,lines,binary_img):
    final_l = np.zeros(len(lines))
    final_sym = np.zeros(len(unKnownSymbols))
    lines_minrects = img_process.getMinRect(lines)
    syms_minrects = img_process.getMinRect(unKnownSymbols)
    leqs = []
    geps = []
    for i in range(len(lines)):
        lx,ly,lw,lh = lines_minrects[i]
        for j in range(len(unKnownSymbols)):
            sx,sy,sw,sh = syms_minrects[j]
            if np.abs(lx-sx)<np.min((lx,sx))/3 and np.abs(lw-sw)<np.min((lw,sw))/3 and 0<ly-sy-sh<sh:
                sym_image = img_process.getSymbolImg(unKnownSymbols[j], (sx,sy,sw,sh), global_config.IMG_SIZE, binary_img)
                sym_image = sym_image.reshape((sym_image.shape[0] * sym_image.shape[1],))
                candidata_syms, candidata_p = symClasser.predict([sym_image]*3)
                can_sym_res = [item for sublist in candidata_syms for item in sublist]
                # candidata_p = lambda multiple_list: [item for sublist in candidata_p for item in sublist]
                print("合并大于等于\小于等于:",can_sym_res)
                if "greater" in can_sym_res:
                    geps.append(copy.deepcopy(lines[i] + unKnownSymbols[j]))
                    final_l[i]=1
                    final_sym[j]=1
                    break
                if "less" in can_sym_res:
                    leqs.append(copy.deepcopy(lines[i] + unKnownSymbols[j]))
                    final_l[i] = 1
                    final_sym[j] = 1
                    break
    res_lines = []
    for i in range(len(lines)):
        if final_l[i]==0:
            res_lines.append(lines[i])

    res_unKnownSymbols = []
    for i in range(len(unKnownSymbols)):
        if final_sym[i]==0:
            res_unKnownSymbols.append(unKnownSymbols[i])
    return res_unKnownSymbols,res_lines,geps,leqs


def getSegmentation(binary_img,ori_image=None):
    # ungerSmooth(binary_img)
    # plt.imshow(binary_img, cmap="Greys", interpolation="None")
    # plt.show()
    # maybeImpurity(smooth_binary_img)
    binary_img = skeletonize(binary_img).astype(np.int32)  # skeletonize函数用于将图像轮廓转为宽度为1个像素的图像
    plt.imshow(binary_img, cmap="Greys", interpolation="None")
    plt.show()
    unKnownSymbols = getSymbols(binary_img)
    # # 合并被意外分隔开的字符
    unKnownSymbols,mybe_sqrts,mybe_ints = mergeSplitSymbol(unKnownSymbols,binary_img)
    # # # 合并等于号
    unKnownSymbols,equals,equivs = mergeEqual(unKnownSymbols)
    # # 合并除法符号
    unKnownSymbols,divs,lines,points,verLines = mergeDiv(unKnownSymbols)
    # 合并大于等于和小于等于符号
    unKnownSymbols,lines,geps,leqs = mergeLeg_Gep(unKnownSymbols,lines,binary_img)
    # # 合并i或j
    unKnownSymbols, points, iSyms, jSyms = mergeiandj(unKnownSymbols, points, binary_img)
    minRects = img_process.getMinRect(unKnownSymbols+divs+iSyms+jSyms+lines+equals+mybe_sqrts+mybe_ints+points+verLines+geps+leqs)
    # # print("smooth_binary_img_shape:",binary_img.shape)
    # # print("symbols_len:",len(symbols))
    # # print(symbols[0])
    drawSymbolsRect(minRects,ori_image)

    # for i in range(len(unKnownSymbols)):
    #     symbol_img = img_process.getSymbolImg(unKnownSymbols[i], minRects[i], global_config.IMG_SIZE, binary_img)
    #     symbol_img = symbol_img.reshape((symbol_img.shape[0]*symbol_img.shape[1],))
    #     syns,ps = symClasser.predict([symbol_img])
    #     print("candidata:",syns,"p:",ps)
    #     plt.imshow(symbol_img.reshape((global_config.IMG_SIZE,global_config.IMG_SIZE)), cmap="Greys", interpolation="None")
    #     plt.show()

    knownSymbols = {"equal":equals,"equiv":equivs,"div":divs,"line":lines,"verLine":verLines,"i":iSyms,"j":jSyms,"sqrt":mybe_sqrts,"int":mybe_ints,"gep":geps,"leq":leqs}
    knownSymbolsInfo = []
    for name,syms in knownSymbols.items():
        if len(syms)<1:
            continue
        symsRects = img_process.getMinRect(syms)
        for i in range(len(syms)):
            symsInfo = {}
            symsInfo["name"] = [name]
            symsInfo["probability"] = [1]
            symsInfo["minRect"] = symsRects[i]
            symsInfo["symPixel"] = syms[i]
            knownSymbolsInfo.append(symsInfo)

    return unKnownSymbols,knownSymbolsInfo,points

if __name__ == "__main__":
    ori_img = cv2.imread("C:/Users/Pengr/Desktop/Testfloat2.png")
    binary_img = img_process.getBinaryImg(ori_img)
    getSegmentation(binary_img,ori_img)
    # plt.imshow(binary_img, cmap="Greys", interpolation="None")
    # plt.show()
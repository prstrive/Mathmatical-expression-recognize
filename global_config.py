IMG_SIZE = 100
DATASET1_SHAPES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
              "u", "v", "w", "x", "y", "z","i_low_vl","j_low_vl",
              "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
              "U", "V", "W", "X", "Y", "Z",
              "0","1","2","3","4","5","6","7","8","9",
              "line", "+", "=", "(", ")", "{", "/", r"\div", r"\geq", r"\infty", r"\int", r"\leq", "<", ">",
              r"\sqrt", r"\sum", r"\times", r"\to",
              r"\alpha", r"\beta", r"\delta", r"\eta", r"\gamma", r"\lambda", r"\mu", r"\omega", r"\phi", r"\pi",
              r"\theta", r"\varphi", r"\zeta"]

INFTYCDB_3_SHAPES = ["a", "b", "c", "d", "e", "f", "g", "h", "l", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                   "s", "t","u", "v", "w", "x", "y", "z",
                   "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                   "S", "T","U", "V", "W", "X", "Y", "Z",
                   "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                   "plus", "RightPar", "LeftPar", "LeftBrace", "slash", "infty", "int", "less",
                   "greater",
                   "sqrt", "Sigma", "times", "rightarrow",
                   "alpha", "beta", "delta", "eta", "gamma", "lambda", "mu", "omega","phi", "pi","theta", "varphi", "zeta"]

# line可为减号、分号等
# point没有分配类型
INFTYCDB_3_SHAPES_INFO = {"a":{"type":"central","filedType":"ss","latex":"a"}, "b":{"type":"ascendent","filedType":"ss","latex":"b"}, "c":{"type":"central","filedType":"ss","latex":"c"},
                          "d":{"type":"ascendent","filedType":"ss","latex":"d"}, "e":{"type":"central","filedType":"ss","latex":"e"}, "f":{"type":"doubleWay","filedType":"ss","latex":"f"},
                          "g":{"type":"descendent","filedType":"ss","latex":"g"}, "h":{"type":"ascendent","filedType":"ss","latex":"h"}, "i":{"type":"ascendent","filedType":"ss","latex":"i"},
                          "j":{"type":"doubleWay","filedType":"ss","latex":"j"}, "k":{"type":"ascendent","filedType":"ss","latex":"k"}, "l":{"type":"ascendent","filedType":"ss","latex":"l"},
                          "m":{"type":"central","filedType":"ss","latex":"m"}, "n":{"type":"central","filedType":"ss","latex":"n"}, "o":{"type":"central","filedType":"ss","latex":"o"}, "p":{"type":"descendent","filedType":"ss","latex":"p"},
                          "q":{"type":"descendent","filedType":"ss","latex":"q"}, "r":{"type":"central","filedType":"ss","latex":"r"},"s":{"type":"central","filedType":"ss","latex":"s"}, "t":{"type":"ascendent","filedType":"ss","latex":"t"},
                          "u":{"type":"central","filedType":"ss","latex":"u"}, "v":{"type":"central","filedType":"ss","latex":"v"}, "w":{"type":"central","filedType":"ss","latex":"w"}, "x":{"type":"central","filedType":"ss","latex":"x"},
                          "y":{"type":"descendent","filedType":"ss","latex":"y"}, "z":{"type":"central","filedType":"ss","latex":"z"},
                          "A":{"type":"ascendent","filedType":"ss","latex":"A"}, "B":{"type":"ascendent","filedType":"ss","latex":"B"}, "C":{"type":"ascendent","filedType":"ss","latex":"C"}, "D":{"type":"ascendent","latex":"ascendent","filedType":"ss","latex":"D"},
                          "E":{"type":"ascendent","filedType":"ss","latex":"E"}, "F":{"type":"ascendent","filedType":"ss","latex":"F"}, "G":{"type":"ascendent","filedType":"ss","latex":"G"}, "H":{"type":"ascendent","filedType":"ss","latex":"H"},
                          "I":{"type":"ascendent","filedType":"ss","latex":"I"}, "J":{"type":"ascendent","filedType":"ss","latex":"J"}, "K":{"type":"ascendent","filedType":"ss","latex":"K"}, "L":{"type":"ascendent","filedType":"ss","latex":"L"},
                          "M":{"type":"ascendent","filedType":"ss","latex":"M"}, "N":{"type":"ascendent","filedType":"ss","latex":"N"}, "O":{"type":"ascendent","filedType":"ss","latex":"O"}, "P":{"type":"ascendent","filedType":"ss","latex":"P"},
                          "Q":{"type":"ascendent","filedType":"ss","latex":"Q"}, "R":{"type":"ascendent","filedType":"ss","latex":"R"}, "S":{"type":"ascendent","filedType":"ss","latex":"S"}, "T":{"type":"ascendent","filedType":"ss","latex":"T"},
                          "U":{"type":"ascendent","filedType":"ss","latex":"U"}, "V":{"type":"ascendent","filedType":"ss","latex":"V"}, "W":{"type":"ascendent","filedType":"ss","latex":"W"}, "X":{"type":"ascendent","filedType":"ss","latex":"X"},
                          "Y":{"type":"ascendent","filedType":"ss","latex":"Y"}, "Z":{"type":"ascendent","filedType":"ss","latex":"Z"},
                          "zero":{"type":"ascendent","filedType":"ss","latex":"0"}, "one":{"type":"ascendent","filedType":"ss","latex":"1"}, "two":{"type":"ascendent","filedType":"ss","latex":"2"}, "three":{"type":"ascendent","filedType":"ss","latex":"3"},
                          "four":{"type":"ascendent","filedType":"ss","latex":"4"}, "five":{"type":"ascendent","filedType":"ss","latex":"5"}, "six":{"type":"ascendent","filedType":"ss","latex":"6"}, "seven":{"type":"ascendent","filedType":"ss","latex":"7"},
                          "eight":{"type":"ascendent","filedType":"ss","latex":"8"}, "nine":{"type":"ascendent","filedType":"ss","latex":"9"},
                          "plus":{"type":"central2","filedType":"none","latex":"+"}, "RightPar":{"type":"central3","filedType":"sp","latex":")"}, "LeftPar":{"type":"central2","filedType":"none","latex":"("}, "LeftBrace":{"type":"central","filedType":"r","latex":"{"},
                          "slash":{"type":"central","filedType":"none","latex":"/"}, "infty":{"type":"central","filedType":"none","latex":"\\infty"}, "int":{"type":"central","filedType":"rbsp","latex":"\\int"}, "less":{"type":"central","filedType":"none","latex":"<"},
                          "greater":{"type":"central","filedType":"none","latex":">"},"sqrt":{"type":"central2","filedType":"spclb","latex":"\\sqrt"},"Sigma":{"type":"central","filedType":"ab","latex":"\\sum"}, "times":{"type":"central","filedType":"none","latex":"\\times"},
                          "rightarrow":{"type":"central","filedType":"none","latex":"\\to"},
                          "alpha":{"type":"central","filedType":"ss","latex":"\\alpha"}, "beta":{"type":"doubleWay","filedType":"ss","latex":"\\beta"}, "delta":{"type":"ascendent","filedType":"ss","latex":"\\delta"}, "eta":{"type":"descendent","filedType":"ss","latex":"\\eta"},
                          "gamma":{"type":"descendent","filedType":"ss","latex":"\\gamma"}, "lambda":{"type":"ascendent","filedType":"ss","latex":"\\lambda"}, "mu":{"type":"descendent","filedType":"ss","latex":"\\mu"}, "omega":{"type":"central","filedType":"ss","latex":"\\omega"},
                          "phi":{"type":"doubleWay","filedType":"ss","latex":"\\phi"}, "pi":{"type":"central","filedType":"ss","latex":"\\pi"},"theta":{"type":"ascendent","filedType":"ss","latex":"\\theta"}, "varphi":{"type":"descendent","filedType":"ss","latex":"\\varphi"}, "zeta":{"type":"doubleWay","filedType":"ss","latex":"\\zeta"},
                          "equal":{"type":"central2","filedType":"none","latex":"="},"equiv":{"type":"central2","filedType":"none","latex":"\\equiv"},"line":{"type":"central2","filedType":"ab"},"div":{"type":"central2","filedType":"none","latex":"\\div"},"point":{"latex":".","filedType":"none"},
                          "lim":{"type":"ascendent","filedType":"b","latex":"\\lim"},"cos":{"type":"central","filedType":"none","latex":"\\cos"},"sin":{"type":"central","filedType":"none","latex":"\\sin"},"verLine":{"type":"central","filedType":"sp","latex":"|"},
                          "leq":{"type":"central2","filedType":"none","latex":"\\leq"},"gep":{"type":"central2","filedType":"none","latex":"\\gep"}
                          }

FILEDTYPE_INFO = {"ss":["super","sub"],"none":[],"ab":["above","bottom"],"b":["bottom"],"spclb":["super","content"],"rbsp":["rightBottom","super"],"r":["right"],"sp":["super"]}

TYPE_INFO = {"central":{"centroid":(1,1,0.5,0.4),"subThreshold":(0,1,0,0.8),"superThreshold":(0,1,0,0.2)},
             "central2":{"centroid":(1,1,0.5,0.5),"subThreshold":(0,1,0,0.8),"superThreshold":(0,1,0,0.2)},
             "central3":{"centroid":(1,1,0.5,0.5),"subThreshold":(0,1,0,0.8),"superThreshold":(0,1,0,0.3)},
             "ascendent":{"centroid":(1,1,0.5,0.6),"subThreshold":(0,1,0,0.8),"superThreshold":(0,1,0,0.3)},
             "descendent":{"centroid":(1,1,0.5,0.3),"subThreshold":(0,1,0,0.4),"superThreshold":(0,1,0,0.1)},
             "doubleWay":{"centroid":(1,1,0.5,0.5),"subThreshold":(0,1,0,0.8),"superThreshold":(0,1,0,0.2)}
             }

NUMBER = ["zero","one","two", "three", "four", "five", "six", "seven", "eight", "nine"]

OPERAND = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
           "s", "t","u", "v", "w", "x", "y", "z","A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
           "K", "L", "M", "N", "O", "P", "Q", "R","S", "T","U", "V", "W", "X", "Y", "Z","zero", "one",
           "two", "three", "four", "five", "six", "seven", "eight", "nine","alpha", "beta", "delta",
           "eta", "gamma", "lambda", "mu", "omega","phi", "pi","theta", "varphi", "zeta","infty"]

OPERATOR = ["plus", "slash", "less","greater","times", "rightarrow","div","gep","leq","<",">"]

INFTYCDB_3_SHAPES_num = {'a': 2158, 'b': 1749, 'c': 1604, 'd': 1995, 'e': 2010, 'f': 1524, 'g': 1524, 'h': 2164, 'i': 3370, 'j': 954, 'k': 1564, 'l': 2868, 'm': 1788, 'n': 3079, 'o': 1515, 'p': 1675, 'q': 1093, 'r': 1964, 's': 1993, 't': 2061, 'u': 1957, 'v': 1331, 'w': 1305, 'x': 1632, 'y': 1429, 'z': 1250, 'A': 1871, 'B': 1443, 'C': 1833, 'D': 1666, 'E': 1581, 'F': 1407, 'G': 1306, 'H': 1434, 'I': 1855, 'J': 1076, 'K': 1251, 'L': 1504, 'M': 1320, 'N': 1490, 'O': 1341, 'P': 1788, 'Q': 953, 'R': 2038, 'S': 1751, 'T': 1724, 'U': 1218, 'V': 1027, 'W': 957, 'X': 823, 'Y': 834, 'Z': 892, 'zero': 1453, 'one': 2717, 'two': 2054, 'three': 1675, 'four': 1474, 'five': 1259, 'six': 1250, 'seven': 1074, 'eight': 1119, 'nine': 1050, 'plus': 1632, 'RightBracket': 933, 'LeftBracket': 949, 'slash': 1338, 'infty': 1365, 'int': 1194, 'less': 1016, 'greater': 781, 'sqrt': 1200, 'Sigma': 1506, 'times': 834, 'rightarrow': 1666, 'alpha': 1626, 'beta': 1217, 'delta': 1219, 'eta': 796, 'gamma': 1204, 'lambda': 1166, 'mu': 1166, 'omega': 861, 'phi': 742, 'pi': 1322, 'theta': 1230, 'varphi': 846, 'zeta': 605}

ALL_SHAPES = INFTYCDB_3_SHAPES

SHAPES_LEN = len(ALL_SHAPES)

INFTYCDB_3_SHAPES_LEN = 128439

# 细化使用的数组
XIHUA_ARRAY = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
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
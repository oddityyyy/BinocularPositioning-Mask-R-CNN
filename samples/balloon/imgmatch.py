# imread按照默认的color模式读入的为BGR格式
import cv2
import re
import numpy as np
import random

# 用于实现匹配，其中i是一个计数器，color是一个随机的颜色[B,G,R]
def match_image(target, tpl, color: list, i: int):
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
    # opencv的六种匹配方法我全试过了，只有后面带有NORMED的三种用起来效果比较好
    th, tw = tpl.shape[:2]  # th,tw是模板的高度和宽度
    md = methods[2]
    result = cv2.matchTemplate(target, tpl, md)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # 对前两种匹配方法，匹配成功的苹果其max_val一般大于0.9
#    print(
#        f'min_val:{min_val},max_val:{max_val},min_loc:{min_loc},max_loc{max_loc}')
#    print(min_val)

    if md == cv2.TM_SQDIFF_NORMED:
        tl = min_loc
        # print(min_val)
        if min_val > 0.15:
            return -1
    else:
        tl = max_loc
        if max_val < 0.85:
            return -1
    # tl,br是匹配到的最终位置，tl是最左上角的点，br是最右下角的点，一定要注意tl和br都是tuple类型不是list类型
    br = (tl[0] + tw, tl[1] + th)
    cv2.rectangle(target, tl, br, color, 2)  # 画图函数会直接改变输入图像，而不需要再把值赋回给输入的图像
    # print(f'tl={tl},br={br}')
    return list(tl+br)


def match(left, right, bbox):#模板匹配
    h, w = right.shape[:2]
    length = int(len(bbox) / 4)
    # print('template在target中的匹配结果：')
    matchList = []
    for i in range(0, length):
        flag = i * 4
        flag2 = 0
        color = [random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)]
        template = left[bbox[flag+1]:bbox[flag+3],
                        bbox[flag]:bbox[flag+2]]
        #cv2.imwrite(str(i)+'.png', template)
        # 此处有一个需要注意的地方，template是从target中裁剪下来的一个对象，但是对template进行更改的话target中对应的区域也会受影响
        # 这样很方便但是也必须要注意，因此如果只是要裁剪图片，那么应该img=img[,]，也就是前后对象名字要一样，不然应该会浪费一点内存
        # 当然如果裁剪下来之后就不用被裁减的对象了的话那也无所谓，毕竟也就浪费一丢丢内存
        if bbox[flag + 1] < 10:
            target = right[0:bbox[flag + 3] + 10, 0:bbox[flag + 2]]
            flag2 = 1
        elif bbox[flag + 3] > h - 10:
            target = right[bbox[flag + 1] - 10:h, 0:bbox[flag + 2]]
            flag2 = 2
        else:
            target = right[bbox[flag + 1] -
                           10:bbox[flag + 3] + 10, 0:bbox[flag + 2]]
        matchResult = match_image(target, template, color, i)
        if matchResult == -1:  # 未匹配到.则在原视图的对应位置画斜线，并且matchList追加-1,-1,-1,-1
            cv2.line(left, (bbox[flag], bbox[flag+1]),
                     (bbox[flag + 2], bbox[flag + 3]), [0, 0, 255], 5, cv2.LINE_AA)
            matchList += [-1, -1, -1, -1]
            continue
        else:  # 匹配到，则在原视图对应位置画框，并在另一个视图对应位置也画框，matchList追加xmin,ymin,xmax,ymax
            cv2.rectangle(left, (bbox[flag], bbox[flag+1]),
                          (bbox[flag + 2], bbox[flag + 3]), color, 2)
            # 此处是为了将匹配到的坐标从相对于target的坐标转换为相对于视图的绝对坐标
            if flag2 == 0:
                matchResult[1] += bbox[flag + 1]-10
                # 转换后的绝对坐标是matchResult=xmin,ymin,xmax,ymax
                matchResult[3] += bbox[flag + 1] - 10
            elif flag2 == 1:
                matchResult[1] += bbox[flag + 1]
                matchResult[3] += bbox[flag + 1]
            elif flag2 == 2:
                matchResult[1] += bbox[flag + 1]-10
                matchResult[3] += bbox[flag + 1]-10
            matchList += matchResult
#            print(bbox[flag], bbox[flag + 1], bbox[flag + 2], bbox[flag + 3])
#            print('匹配结果')
#            print(matchResult)
#            print('')
    #cv2.imshow('left', left)
    #cv2.imshow('right', right)
    #cv2.waitKey(0)
    #cv2.imwrite('lMatchResult.png', left)
    #cv2.imwrite('rMatchResult.png', right)
    return matchList


'''    if input('save result as .png? y/n ') == 'y':
        cv2.imwrite('lMatchResult.png', left)
        cv2.imwrite('rMatchResult.png', right)'''
# print(f'左图标签：x:{lx}  y:{ly}')
# print(f'右图标签：x:{rx}  y:{ry}')


def mrbbox_match(mbbox,rbbox):#MaskRCNN实例匹配，lbbox、mbbox与rbbox的匹配
    result=[]
    RBBOX=[]
    
    for i in range(int(len(mbbox)/4)):
        index=0
        min=1000
        mflag=4*i
        if mbbox[mflag]==-1:
            result.append(-1)
            RBBOX=RBBOX+[-1,-1,-1,-1]
            continue
        for j in range(int(len(rbbox)/4)):
            rflag=4*j
            pre=abs(mbbox[mflag]-rbbox[rflag])+abs(mbbox[mflag+1]-rbbox[rflag+1])+abs(mbbox[mflag+2]-rbbox[rflag+2])+abs(mbbox[mflag+3]-rbbox[rflag+3])
            if min>pre:
                min=pre
                index=rflag
        if min>100:
            result.append(-1)
            RBBOX=RBBOX+[-1,-1,-1,-1]
        else:
            #print(index)
            #print(abs(mbbox[mflag]-rbbox[index])+abs(mbbox[mflag+1]-rbbox[index+1])+abs(mbbox[mflag+2]-rbbox[index+2])+abs(mbbox[mflag+3]-rbbox[index+3]))
            #print(mbbox[mflag:mflag+4])
            #print(rbbox[index:index+4])
            #print()
            result.append(int(index/4))
            #rbbox[index:index+4]=10000,10000,10000,10000
            RBBOX=RBBOX+rbbox[index:index+4]
    return result,RBBOX
#最终返回的列表result其元素由每组mbbox对应的rmasks的下标组成
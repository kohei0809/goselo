import cv2
import random
import numpy as np
import math
import scipy

def printImage(image):
    cv2.imshow('map',image)
    cv2.waitKey()
    
def printColorImage(image):
    m = image.shape[0]
    n = image.shape[1]
    color_image = np.zeros((m, n, 3))
    
    #壁以外を白にする
    for i in range(m):
        for j in range(n):
            if image[i][j] == 1:
                #壁は黒色
                for k in range(3):
                    color_image[i][j][k] = 0
            else:
                #壁以外は白色
                for k in range(3):
                    color_image[i][j][k] = 255
                    
    printImage(color_image)
    
if __name__ == '__main__':
    #GOSELOマップの変換テスト
    env = 'default'
    map_path = '../map_images/' + env + '.png'
    change_color = True

    target_imsize = (224, 224)

    ########################
    # load map imformation #
    ########################
    im_map = cv2.imread(map_path) #[縦][横][BGR]
    #print(im_map)
    #マップを表示
    printImage(im_map)
    
    #print(len(im_map>100))
    im_map[im_map > 100] = 255 #change gray to white
    n = im_map.shape[1] #horizontal size of the map
    m = im_map.shape[0] #vertical size of the map
    print("n=" + str(n) + ", m=" + str(m))

    the_map = []
    the_map_pathlog = []
    the_map_expand = []
    row = [0] * n
    row2 = [0] * n
    row3 = [0] * n
    for i in range(m):
        the_map.append(list(row))
        the_map_pathlog.append(list(row2))
        the_map_expand.append(list(row3))
    
    ################################
    # set start and goal locations #
    ################################
    while True:
        xA = random.randint(0, n-1)
        yA = random.randint(0, m-1)
        xB = random.randint(0, n-1)
        yB = random.randint(0, m-1)
        if (im_map[yA][xA][0] > 100) and (im_map[yB][yB][0] > 100):
            break
    
 
    xA=180; yA=70; xB=150; yB=100;    
    start = (xA, yA)
    goal = (xB, yB)
    print("start:" + str(start))
    print("goal:" + str(goal))

    if change_color:
        #startを赤、goalを青
        im_map_ = im_map.copy()
        cv2.circle(im_map_, (xA, yA), 3, (0, 0, 255), -1)
        cv2.circle(im_map_, (xB, yB), 3, (255, 0, 0), -1)
        printImage(im_map_)
        
    ##################
    # expand the map #
    ##################
    #白色の領域を収縮
    im_map_expand = cv2.erode(im_map, np.ones((2, 2), np.uint8), 1)
    #黒を1, 白を0にする
    ret, the_map = cv2.threshold(im_map, 100, 1, cv2.THRESH_BINARY_INV)
    #3次元目がRGB表記だったが、白or黒なので、1つだけにする
    the_map = the_map[:, :, 0]

    ###############
    # convert img #
    ###############
    #the_map, start, goal, pathlogをまとめ、一気に回転、拡大をする
    orig_map = np.zeros((4, m, n), dtype=np.float32)
    orig_map[0] = np.array(the_map)
    orig_map[1][yA][xA] = 1
    orig_map[2][yB][xB] = 1
    orig_map[3] = np.array(the_map_pathlog)
    #(m, n, map)
    orig_map = orig_map.transpose((1, 2, 0)) 

    #xA, yAは毎回更新され現在地を示す
    sx = xA; sy = yA; gx = xB; gy = yB
    #現在地とゴールの中点
    mx = (sx + gx) / 2.
    my = (sy + gy) / 2.
    #画面の端からの最大の長さ
    dx_ = max(mx, orig_map.shape[1] - mx)
    dy_ = max(my, orig_map.shape[0] - my)
    im2 = np.zeros((int(dy_ * 2), int(dx_ * 2), orig_map.shape[2]), dtype=np.float32)
    im2[int(dy_-my):int(dy_-my)+orig_map.shape[0], int(dx_-mx):int(dx_-mx)+orig_map.shape[1]] = orig_map
    #壁を2にする(?)
    im2[im2 == 1] = 2
    if gx == sx:
        if gy > sy:
            theta = 90
        else:
            theta = 270
    else:
        theta = math.atan(float(gy-sy) / float(gx-sx)) * 180 / math.pi
        if gx-sx < 0:
            theta = theta + 180
        
    print("theta=" + str(theta))
    print(im2)
    im2 = scipy.ndimage.interpolation.rotate(im2, theta+90)
    #(map, m, n)
    im2 = im2.transpose((2, 0, 1))
    #print(im2.shape)
    printImage(im2[0])
    
    #現在地からゴールまでの距離
    L = int(np.sqrt((gx-sx)*(gx-sx) + (gy-sy)*(gy-sy)))
    #[the_map, the_map_path_log]
    im2 = [im2[0], im2[3]]
    #(224, 224, 6)
    im3 = np.zeros((target_imsize[0], target_imsize[1], 6), dtype=np.uint8)
    #(6, 224, 224)
    im3 = im3.transpose((2, 0, 1))
    l = (L+4, 4*L, 8*L)
    for n_ in range(2):
        for i in range(3):
            im2_ = np.zeros((l[i], l[i]), dtype=np.uint8)
            #int()をつけていいか微妙
            y1 = int(max(0,  (im2[n_].shape[0]-l[i])/2))
            y2 = int(max(0, -(im2[n_].shape[0]-l[i])/2))
            x1 = int(max(0,  (im2[n_].shape[1]-l[i])/2))
            x2 = int(max(0, -(im2[n_].shape[1]-l[i])/2))
            #print("[0], [1], l[i] = " + str(im2[n_].shape[0]) + ", " + str(im2[n_].shape[1]) + ", " + str(l[i]))
        
            #8Lなどがマップからはみ出さないようにする
            dy_ = min(l[i], im2[n_].shape[0])
            dx_ = min(l[i], im2[n_].shape[1])
        
            #print("x1, dx_, y1, dy_ = " + str(x1) + ", " + str(dx_) + ", " + str(y1) + ", " + str(dy_))
            #print("x2, dx_, y2, dy_ = " + str(x2) + ", " + str(dx_) + ", " + str(y2) + ", " + str(dy_))
        
            #該当する箇所だけコピー
            im2_[y2:y2+dy_, x2:x2+dx_] = im2[n_][y1:y1+dy_, x1:x1+dx_]
            print(im2_.shape)
            print(im3[i + n_*3].shape)
            im3[i + n_*3] = cv2.resize(im2_, im3[i + n_*3].shape, interpolation=cv2.INTER_AREA)

    im3 = im3 * 0.5
    im3[(im3 > 0)*(im3 <= 1)] = 1
    print(im3.shape)

    #(224, 224, 6)
    input_map = im3.transpose((1, 2, 0))
    input_map = input_map / 255.

    printColorImage(im3[0])
    printColorImage(im3[1])
    printColorImage(im3[2])

import cv2
import random
import numpy as np
import math
import scipy
import csv
import datetime
import os

from astar import Astar

def printImage(image, tp=None, save=False) -> None:
    if (save == True) and (tp is not None):
        path = dir_name + "/" + str(start) + "->" + str(goal) + tp + ".png"
        cv2.imwrite(path, image)
    cv2.imshow('map',image)
    cv2.waitKey()
    
def printColorImage(image, tp=None, save=False) -> None:
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
                    
    printImage(color_image, tp=tp, save=save)
    
def printPathLog(map, map_pathlog, start, goal):
    #print(len(map_pathlog))
    #print(len(map_pathlog[0]))
    m = len(map_pathlog)
    n = len(map_pathlog[0])
    map_ = np.zeros((m, n, 3))
    #print(map_)
    
    #毛色を緑色にする
    for i in range(m):
        for j in range(n):
            if map [i][j] == 0:
                for k in range(3):
                    map_[i][j][k] = 255
                
            if map_pathlog[i][j] > 0:
                #print(str(j) + "," + str(i))
                cv2.circle(map_, (j, i), 1, (0, 255, 0), -1)
                
    #スタートを赤色、ゴールを青色にする
    cv2.circle(map_, start, 3, (0, 0, 255), -1)
    cv2.circle(map_, goal, 3, (255, 0, 0), -1)
    path = dir_name + "/" + str(start) + "->" + str(goal) + "path.png" 
    cv2.imwrite(path, map_)
    cv2.imshow('pathlog',map_)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def getAstarRoot(map, start, goal, map_pathlog, env, read=False):
    now = datetime.datetime.now()
    print(now)
    
    file_name = ""
    if read == True:
        file_name = "04-23-17-15-39"
        #file_name = "04-23-19-03-48"
        file_name = "./../root/" + env + "/" + file_name + ".csv"
        
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                #print(line)
                x, y = line
                x = int(x)
                y = int(y)
                map_pathlog[y][x] += 1
            
    else:
        print("start:" + str(start))
        print("goal:" + str(goal))
        root_map = Astar(map, start, goal).getRoot()
        print("A* root")
        print(root_map)
    
        now = now.strftime('%m-%d-%H-%M-%S')
        dir_name = "./../root/" + env
        file_name = "./../root/" + env + "/" + now + ".csv"
        
        # ディレクトリが存在しない場合、ディレクトリを作成する
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            for (x, y) in root_map:
                line = [x, y]
                print(line)
                writer.writerow(line)
                map_pathlog[y][x] += 1
    
    return map_pathlog
            
            
if __name__ == '__main__':
    #GOSELOマップの変換テスト
    maps = ["default", "AR0011SR", "AR0205SR", "AR0411SR", "lak303d", "lak308d"]
    env = 5
    map_path = '../map_images/' + maps[env] + '.png'
    change_color = True
    pathlog = True
    csvRead = False

    target_imsize = (224, 224)
    
    dir_name = "./../images/" + maps[env]
    # ディレクトリが存在しない場合、ディレクトリを作成する
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

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
        if (im_map[yA][xA][0] > 100) and (im_map[yB][xB][0] > 100):
                break
    
 
    #default
    #xA=180; yA=70; xB=150; yB=100;    
    #AR0205SR
    #xA=4; yA=100; xB=240; yB=160;    
    start = (xA, yA)
    goal = (xB, yB)
    print("start:" + str(start))
    print("goal:" + str(goal))

    if change_color:
        #startを赤、goalを青
        im_map_ = im_map.copy()
        cv2.circle(im_map_, (xA, yA), 3, (0, 0, 255), -1)
        cv2.circle(im_map_, (xB, yB), 3, (255, 0, 0), -1)
        printImage(im_map_, tp="sg", save=True)
            
    ##################
    # expand the map #
    ##################
    #白色の領域を収縮
    im_map_expand = cv2.erode(im_map, np.ones((2, 2), np.uint8), 1)
    #黒を1, 白を0にする
    ret, the_map = cv2.threshold(im_map, 100, 1, cv2.THRESH_BINARY_INV)
    #3次元目がRGB表記だったが、白or黒なので、1つだけにする
    the_map = the_map[:, :, 0]
    
    if pathlog == True:
        the_map_pathlog = getAstarRoot(the_map, start, goal, the_map_pathlog, maps[env], read=csvRead)
        #print(the_map_pathlog)
        printPathLog(the_map, the_map_pathlog, start, goal)

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
    #print(im2)
    #im2 = scipy.ndimage.interpolation.rotate(im2, theta+90)
    im2 = scipy.ndimage.rotate(im2, theta+90)
    
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
            #print(im2_.shape)
            #print(im3[i + n_*3].shape)
            im3[i + n_*3] = cv2.resize(im2_, im3[i + n_*3].shape, interpolation=cv2.INTER_AREA)

    im3 = im3 * 0.5
    im3[(im3 > 0)*(im3 <= 1)] = 1
    print(im3.shape)

    #(224, 224, 6)
    input_map = im3.transpose((1, 2, 0))
    input_map = input_map / 255.

    printColorImage(im3[0], tp="s", save=True)
    printColorImage(im3[1], tp="m", save=True)
    printColorImage(im3[2], tp="l", save=True)

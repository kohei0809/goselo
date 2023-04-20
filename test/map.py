import cv2
import random

#GOSELOマップの変換テスト
env = 'default'
map_path = '../map_images/' + env + '.png'

########################
# load map imformation #
########################
im_map = cv2.imread(map_path) #[縦][横][RGB]
#print(im_map)
#マップを表示
#cv2.imshow('map',im_map)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

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
    
start = (xA, yA)
goal = (xB, yB)
print("start:" + str(start))
print("goal:" + str(goal))

#startを赤、goalを青
for i in range(-2, 2):
    for j in range(-2, 2):
        #print("i=" + str(i) + ", j=" + str(j))
        if(yA+i >= 0) and (xA+j >= 0) and (yA+i < m) and (xA+j < n):
            if im_map[yA+i][xA+j][2] == 255:
                im_map[yA+i][xA+j][1] = 0
                im_map[yA+i][xA+j][2] = 0
        if(yB+i >= 0) and (xB+j >= 0) and (yB+i < m) and (xB+j < n):
            if im_map[yB+i][xB+j][1] == 255:
                im_map[yB+i][xB+j][0] = 0
                im_map[yB+i][xB+j][1] = 0

    

cv2.imshow('map',im_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

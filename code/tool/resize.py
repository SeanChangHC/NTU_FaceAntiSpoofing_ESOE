import os
import cv2

height, width = 108, 192
origin_root = '/home/fas3/example-img/OULU_backup/Train'
files = os.listdir(origin_root)
goal_root = '/home/fas3/example-img/OULU/train_resize'

for f in files:
    file = os.path.join(origin_root, f)
    image = cv2.imread(file)
    resize_image = cv2.resize(image, (height, width))
    cv2.imwrite(file.replace('.jpg', '.resize.jpg').replace(origin_root, goal_root), resize_image)
    print(file)
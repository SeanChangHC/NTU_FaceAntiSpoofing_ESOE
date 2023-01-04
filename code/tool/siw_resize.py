import cv2
import os
origin_root = '/home/fas3/example-img/SiW60_protoco1/Test/all'
files = os.listdir(origin_root)
goal_root = '/home/fas3/example-img/SiW/protocol1/test'


scale_percent = 20 # percent of original size


for f in files:
    file = os.path.join(origin_root, f)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ',img.shape)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resize_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resize_image.shape)
    # cv2.imshow("Resized image", resize_image)
    cv2.imwrite(file.replace('.jpg', '.resize.jpg').replace(origin_root, goal_root), resize_image)
    
    print(file)



 
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import os
import cv2

def selectFrameNumber(totalFrame):
    number = []
    number.append(totalFrame//4)
    number.append(totalFrame//4 * 2)
    number.append(totalFrame//4 * 3)
    return number
    


root = '/home/fas3/img/OULU/Test_files/avi'
files = os.listdir(root)
files.sort()

for i, f in enumerate(files):
    cap = cv2.VideoCapture(os.path.join(root, f))
    totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selectedFrame = selectFrameNumber(totalFrame)
    
    for j, frame_no in enumerate(selectedFrame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret == True:
            imgName = f.replace('.avi', f'.0{j+1}.jpg')
            cv2.imwrite(imgName, frame)
            print(f'{i}/{len(files)}')
        else:
            print(f'processing {f} is error!')
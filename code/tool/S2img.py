import os
import cv2

def selectFrameNumber(totalFrame):
    number = []
    for i in range(60):
        number.append(i)
    return number
    


input_folder = '/home/fas3/img/SiW/Train/spoof/mov/'
output_folder = '/home/fas3/example-img/SiW60/Train/spoof/'
files = os.listdir(input_folder)
files.sort()


for i, f in enumerate(files):
    cap = cv2.VideoCapture(os.path.join(input_folder, f))
    totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selectedFrame = selectFrameNumber(totalFrame)
    
    for j, frame_no in enumerate(selectedFrame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret == True:
            imgName = f.replace('.mov', f'.spoof.0{j+1}.jpg')
            cv2.imwrite(os.path.join(output_folder, imgName), frame)
        else:
            print(f'processing {f} is error!')
    print(f'{i+1}/{len(files)}')
    
        

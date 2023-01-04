import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
input_folders = ['/home/fas3/example-img/OULU_backup/Test/']
output_folders = ['/home/fas3/example-img/OULU_backup/yolos_test/']

def yolo_tags(input_folder, output_folder):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    files = os.listdir(input_folder)
    min_size = 100
    create_img = False

    for f in files:
        img_path = os.path.join(input_folder, f)
        txt_path = os.path.join(output_folder, f.replace('.jpg', '.txt'))
        face_img_path = os.path.join(output_folder, f)

        img = cv2.imread(img_path)
        img_argument = [0]*5
        img_height, img_weight, *_ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
        print(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            if w > int(min_size) or h > int(min_size) :
                if f.replace('.', '_').split('_')[1] == 'live':
                    img_argument[0] = '1'
                else:
                    img_argument[0] = '0'
                img_argument[1] = str(round((x+w/2) / img_weight, 6))
                img_argument[2] = str(round((y+h/2) / img_height, 6))
                img_argument[3] = str(round(w / img_weight, 6))
                img_argument[4] = str(round(h / img_height, 6))
                print(img_argument)
                print(txt_path)
                with open(txt_path, 'w') as o_path:
                    print(f'{txt_path} saved!')
                    print(' '.join(img_argument), file=o_path)

                if create_img:
                    faces = img[y:y + h, x:x + w]
                    cv2.imwrite(face_img_path, faces)
                break
        # cv2.imshow('img', img)
        # print(f)
        # print("長:{0} 寬:{1}".format(w, h))
        # print(img_argument)

for i in range(len(input_folders)):
    yolo_tags(input_folders[i], output_folders[i])
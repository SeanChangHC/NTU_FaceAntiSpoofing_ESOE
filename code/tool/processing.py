import os
import shutil

input_folder = '/Users/shibasalmon/Code/python-test/SiW/Train/'
output_folder = '/Users/shibasalmon/Code/python-test/SiW_Yolos/Train/'
# untrans_folder = '/Users/shibasalmon/Code/python-test/OULU/Yolos/nti/'
trans_folder = '/Users/shibasalmon/Code/python-test/SiW_Trans_IMG/Train/'

img = os.listdir(input_folder)
img.sort()
txt = os.listdir(output_folder)
txt.sort()

untrans = []

for i in range(len(img)):
    if img[i].replace('.jpg', '.txt') in txt:
        print(img[i])
        shutil.copyfile(os.path.join(input_folder, img[i]), os.path.join(trans_folder, img[i]))
        untrans.append(img[i])

print(len(untrans))
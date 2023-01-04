
import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt


def plot_learning_curve(title_name : str, y_label : str, epoch : list, data : list, img_path = './'):
    font1 = {'family':'serif','color':'black','size':20}
    font2 = {'family':'serif','color':'blue','size':15}
    plt.title(title_name, fontdict = font1)
    plt.plot(epoch, data, color='red') 
    plt.xlabel('epoch', fontdict = font2)
    plt.ylabel(y_label, fontdict = font2)
    plt.savefig(os.path.join(img_path,title_name+'.png'))
    plt.show()
    
    
    
epoch_list = [x  for x in range(20)]
overall_loss = [0.020033085718750954, 0.004151380155235529, 0.001745621208101511, 0.0023655351251363754, 0.0012997284065932035, 0.001685180002823472, 0.001307526370510459, 0.0009066188358701766, 0.0006852815859019756, 0.0005883325939066708, 0.0007719451677985489, 0.000571735727135092, 0.0006077805883251131, 0.0005651205428875983, 0.0002938863472081721, 0.0002481239498592913, 0.0007791324751451612, 0.00029519505915232003, 0.0002777355839498341, 0.00032647105399519205]
overall_val_loss = [0.07549197971820831, 0.013289607129991055, 0.003921381197869778, 0.003945109900087118, 0.002589741488918662, 0.0045718601904809475, 0.0013180607929825783, 0.002323322929441929, 0.002014326862990856, 0.0007006050436757505, 0.0005323708755895495, 0.0005935541121289134, 0.0005491782794706523, 0.0004190678591839969, 0.00023239011352416128, 0.00034451764076948166, 0.0002090443595079705, 0.0002522766008041799, 0.0002044893044512719, 0.00023568462347611785]
plot_learning_curve(title_name = 'Validation Loss', y_label = 'Loss', epoch = epoch_list, data = overall_val_loss)
# plot_learning_curve(title_name = 'Training Loss', y_label = 'Loss', epoch = epoch_list, data = overall_loss)
print(len(overall_val_loss))
print(len(epoch_list))
# print(epoch_list)
from sklearn.metrics import confusion_matrix 
import torch
import torch.nn as nn

import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
from cfg import LeNet_cfg as cfg

img_root = cfg['img_root']
best_model = cfg['best_model']
def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    # APCER = false_negatives / (true_positives + false_negatives)
    # BPCER = false_positives / (false_positives + true_negatives)
    # ACER = (APCER + BPCER) / 2
    # return APCER, BPCER, ACER
    return true_positives, false_positives, true_negatives, false_negatives

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
        
        
def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print("End of saving !!!")


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")



## TO DO ##
def plot_learning_curve(title_name : str, y_label : str, epoch : list, data : list, img_path = img_root):
    font1 = {'family':'serif','color':'black','size':20}
    font2 = {'family':'serif','color':'blue','size':15}
    plt.title(title_name, fontdict = font1)
    plt.plot(epoch, data, color='red') 
    plt.xlabel('epoch', fontdict = font2)
    plt.ylabel(y_label, fontdict = font2)
    plt.savefig(os.path.join(img_path, title_name+'.png'))
    plt.show()
    

def train(model, train_loader, val_loader, num_epoch, log_path, save_path, device, criterion, scheduler, optimizer):
    start_train = time.time()
    #每一個epoch的loss / acc / ACER
    overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_acc = np.zeros(num_epoch ,dtype = np.float32)
    overall_ACER = np.zeros(num_epoch ,dtype = np.float32)
    overall_val_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_val_acc = np.zeros(num_epoch ,dtype = np.float32)
    overall_val_ACER = np.zeros(num_epoch ,dtype = np.float32)
    
    best_acc = 0
    best_ACER = 1
    for i in range(num_epoch):
        print(f'epoch = {i}')
        # epcoch setting
        start_time = time.time()
        train_loss = 0.0 
        corr_num = 0
        
        #宣告 confusion matrix 裡面的四個變數已計算ACER
        TP, FP, TN, FN = 0, 0, 0, 0
        TP_val, FP_val, TN_val, FN_val = 0, 0, 0, 0
        
        # training part
        # start training
        model.train()
        for batch_idx, ( data, label,) in enumerate(tqdm(train_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            label = label.to(device)

            # pass forward function define in the model and get output 
            output = model(data) 

            # calculate the loss between output and ground truth
            loss = criterion(output, label)
            
            # discard the gradient left from former iteration 
            optimizer.zero_grad()

            # calcualte the gradient from the loss function 
            loss.backward()
            
            # if the gradient is too large, we dont adopt it
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # Update the parameters according to the gradient we calculated
            optimizer.step()

            train_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            
            #計算confusion matrix
            true_positives, false_positives, true_negatives, false_negatives = confusion(pred, label)
            
            TN += true_negatives
            FP += false_positives
            TP += true_positives
            FN += false_negatives
            
            # correct if label == predict_label            
            corr_num += (pred.eq(label.view_as(pred)).sum().item())

        # scheduler += 1 for adjusting learning rate later
        scheduler.step()
        
        # averaging training_loss and calculate accuracy
        train_loss = train_loss / len(train_loader.dataset) 
        train_acc = corr_num / len(train_loader.dataset)
        
        #計算training APCER / BPCER / ACER
        train_APCER = FN / (TP + FN)
        train_BPCER = FP / (FP + TN)
        train_ACER = (train_APCER + train_BPCER) / 2      
          
        # record the training loss/acc/ACER
        overall_loss[i], overall_acc[i] , overall_ACER[i] = train_loss, train_acc, train_ACER
        
        ## TO DO ##
        # validation part 
        model.eval()
        valid_loss = []
        valid_accs = []
        valid_ACER = []
        for batch in tqdm(val_loader):
            imgs, labels = batch
            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))
            logits_argmax = logits.argmax(dim=-1)
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            #confusion matrix 
            true_positives_val, false_positives_val, true_negatives_val, false_negatives_val = confusion(logits_argmax.to(device), labels.to(device))
            TN_val += true_negatives_val
            FN_val += false_negatives_val
            TP_val += true_positives_val
            FP_val += false_positives_val
            
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        val_loss = sum(valid_loss) / len(valid_loss)
        val_acc = sum(valid_accs) / len(valid_accs)
        # print(f'TN_val: {TN_val}')
        # print(f'FN_val: {FN_val}')
        # print(f'TP_val: {TP_val}')
        # print(f'FP_val: {FP_val}')
        
        val_APCER = FN_val / (TP_val + FN_val)
        val_BPCER = FP_val / (FP_val + TN_val)
        val_ACER = (val_APCER + val_BPCER) / 2      
        
        # record the validation loss/acc/ACER
        
        overall_val_loss[i], overall_val_acc[i], overall_val_ACER[i]  = val_loss, val_acc, val_ACER

        
        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        #print(f'epoch = {i}')
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}', f'tran ACER = {train_ACER:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' , f'val ACER = {val_ACER:.4f}')
        print('========================\n')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {i}\n', )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        # save model for every epoch 
        #torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{i}.pt'))
        
        # # save the best model if it gain performance on validation set
        # if  val_acc > best_acc:
        #     best_acc = val_acc
        #     torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
        
        
        if val_ACER < best_ACER:
            best_ACER = val_ACER
            torch.save(model.state_dict(), os.path.join(save_path, best_model))


    x = range(0,num_epoch)
    
    epoch_list = [x  for x in range(num_epoch)]
    overall_acc = overall_acc.tolist()
    overall_loss = overall_loss.tolist()
    overall_ACER = overall_ACER.tolist()
    overall_val_acc = overall_val_acc.tolist()
    overall_val_loss = overall_val_loss.tolist()
    overall_val_ACER = overall_val_ACER.tolist()
    
    
    # print(f'epoch_list: \n {epoch_list}')
    # print(f'overall_loss: \n {overall_loss}')
    # print(f'overall_val_acc: \n {overall_val_acc}')
    # print(f'overall_val_loss: \n {overall_val_loss}')
    
    # Plot Learning Curve
    plot_learning_curve(title_name = 'Training Accuracy', y_label = 'Accuracy', epoch = epoch_list, data = overall_acc)
    # plot_learning_curve(title_name = 'Training Loss', y_label = 'Loss', epoch = epoch_list, data = overall_loss)
    # plot_learning_curve(title_name = 'Validation Accuracy', y_label = 'Accuracy', epoch = epoch_list, data = overall_val_acc)
    # plot_learning_curve(title_name = 'Validation Loss', y_label = 'Loss', epoch = epoch_list, data = overall_val_loss)



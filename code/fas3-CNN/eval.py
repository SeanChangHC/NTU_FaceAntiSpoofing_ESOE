from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch 
import json
import argparse
from tqdm import tqdm
import os
from tool import load_parameters
from myModels import  ResNext50
from myDatasets import cifar10_dataset
from cfg import LeNet_cfg as cfg
from glob import glob
from re import search



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



# The function help you to calculate accuracy easily
# Normally you won't get annotation of test label. But for easily testing, we provide you this.
def test_result(test_loader, model, device):
    pred = []
    TP, FP, TN, FN = 0, 0, 0, 0
    cnt = 0
    
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device)
            label = label.to(device)
            # print(f'img size:{img.size()}')
            pred = model(img)
            # print(f'Before argmax:{pred}')
            # print(f'Before pred size: {pred.size()}')
            pred = torch.argmax(pred, axis=1)
            true_positives, false_positives, true_negatives, false_negatives = confusion(pred, label)
            TP += true_positives
            FP += false_positives
            TN += true_negatives
            FN += false_negatives
            # print(f'After argmax:{pred}')
            # print(f'After pred size: {pred.size()}')
            cnt += (pred.eq(label.view_as(pred)).sum().item())
    
    
    APCER = FN / (TP + FN)
    BPCER = FP / (FP + TN)
    ACER = (APCER + BPCER) / 2
    acc = cnt / len(test_loader.dataset)
    return APCER, BPCER, ACER, acc
    
    # return acc

def main():
    ####要修改的地方：測試資料放的位置####
    test_root = cfg['test_root']
    best_model = cfg['best_model']
    ####要修改的地方：測試資料放的位置####
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model type', type=str, default='ResNext50')
    parser.add_argument('--path', help='model_path', type=str, default='./Model/')
    # parser.add_argument('--test_anno', help='annotaion for test image', type=str, default= './p2_data/annotations/public_test_annos.json')
    args = parser.parse_args()
    
    model_type = args.model
    path = args.path + model_type + "/" + best_model
    # test_anno = args.test_anno
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # change your model here 

    ## TO DO ## 
    # Indicate the model you use here
    #ResNext
    if model_type == 'ResNext50':
        model = ResNext50(num_class=2, device=device)
    
    
    # Simply load parameters
    # print(path)
    model.load_state_dict(torch.load(path))
    model.to(device)


   
    
    
    ###抓那個資料夾裡所有的檔案###
    images = glob(test_root+'/*')


    ###將檔案的名稱轉成label存到lables裡面去###
    labels = []
    for root, dirs, files in os.walk(test_root):
        for filename in files:
            # if 'live' in filename:
            #     labels.append(int(1))
            if 'live' in filename:
                labels.append(int(1))
            # elif '_1.' in filename:
            #     labels.append(int(1))
            else:
                labels.append(0)
    
    

    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    test_set = cifar10_dataset(images=images, labels= labels, transform=test_transform, prefix = './p2_data/public_test/')
    #test_set = cifar10_dataset(images=imgs, labels= categories, transform=test_transform, prefix = './p2_data/private_test/')
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    # acc = test_result(test_loader=test_loader, model=model, device=device)
    APCER, BPCER, ACER, acc = test_result(test_loader=test_loader, model=model, device=device)
    
    print("accuracy : ", acc)
    print(f'APCER:{APCER}')
    print(f'BPCER:{BPCER}')
    print(f'ACER:{ACER}')
    
    # print(f'tp: {true_positives}')
    # print(f'tn: {true_negatives}')
    # print(f'fp: {false_positives}')
    # print(f'fn: {false_negatives}')
    

if __name__ == '__main__':
    main()
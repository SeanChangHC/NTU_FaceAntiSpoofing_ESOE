
from sklearn.metrics import confusion_matrix
import torch


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
    print(f'tp1:{true_positives}')
    print(f'fp1: {false_positives}')
    print(f'tn1: {true_negatives}')
    print(f'fn1:{false_negatives}')
    APCER = false_negatives / (true_positives + false_negatives)
    BPCER = false_positives / (false_positives + true_negatives)
    ACER = (APCER + BPCER) / 2
    return APCER, BPCER, ACER

TP, FP, TN, FN = 0, 0, 0, 0
l1 = [1,   1,  1,  0,  0,  0,  0,  0,  0,  0 ]
l2 = [1.0, 1,  0,  0,  1,  0,  0,  1,  1,  1 ]

prediction = torch.tensor(l1)
truth = torch.tensor(l2)

CM = confusion_matrix(prediction, truth)
TN += CM[0][0]
FP += CM[1][0]
TP += CM[1][1]
FN += CM[0][1]
print(f'TP_CM: {TP}')
print(f'FP_CM: {FP}')
print(f'TN_CM: {TN}')
print(f'FN_CM: {FN}')
train_APCER = FN / (TP + FN)
train_BPCER = FP / (FP + TN)
train_ACER = (train_APCER + train_BPCER) / 2  
print(f'CM_APCER:{train_APCER}')
print(f'CM_BPCER:{train_BPCER}')
print(f'CM_ACER:{train_ACER}')




APCER, BPCER, ACER= confusion(prediction, truth)

print(f'APCER:{APCER}')
print(f'BPCER:{BPCER}')
print(f'ACER:{ACER}')
# print(f'fn:{fn}')
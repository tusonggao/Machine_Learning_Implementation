import numpy as np

def auc(labels, pred_p):


label_all = np.random.randint(0,2,[10,1]).tolist()
pred_all = np.random.random((10,1)).tolist()

print(label_all)
print(pred_all)

posNum = len(list(filter(lambda s: s[0] == 1, label_all)))
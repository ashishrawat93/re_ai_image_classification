import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


y_hat = np.load("random_tensor_pred.npy")
y = np.load("random_tensor_labels.npy")

m = abs(y_hat-y)
m[m!=0] = 1
print((y.shape - m.sum())/y.shape)

print(y_hat.shape,y.shape)

#y_pred = cross_val_predict(model,trainDataGlobal,trainLabelsGlobal,cv=kfold)
conf_mat = confusion_matrix(y,y_hat)
print(conf_mat)

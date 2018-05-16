import numpy as np

from sklearn.metrics import confusion_matrix


y_hat = np.load("random_tensor_pred.npy")
y = np.load("random_tensor_labels.npy")

m = abs(y_hat-y)
m[m!=0] = 1
print("Accuracy is : "(y.shape - m.sum())/y.shape)

#y_pred = cross_val_predict(model,trainDataGlobal,trainLabelsGlobal,cv=kfold)
conf_mat = confusion_matrix(y,y_hat)
print(conf_mat)

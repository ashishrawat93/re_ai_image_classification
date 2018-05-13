#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------

# import the necessary packages
import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
import mahotas



# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=2)))
fixed_size = tuple((500, 500))
# variables to hold the results and names
results = []
names = []
scoring = "accuracy"

# import the feature vector and trained labels
h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()



# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

(trainDataGlobal2, testDataGlobal2, trainLabelsGlobal2, testLabelsGlobal2) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=0.001,
                                                                                          random_state=seed)


'''
trainDataGlobal = np.array(global_features)
trainLabelsGlobal = np.array(global_labels)
trainLabelsGlobal = np.ones(1)
testLabelsGlobal = np.ones(1)
'''
print(global_features.shape,trainDataGlobal.shape)
print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
#    print(trainDataGlobal2.shape)
    cv_results = cross_val_score(model, trainDataGlobal2, trainLabelsGlobal2, cv=kfold, scoring=scoring)
#    cv_results = cross_val_score(model,np.array(global_features) , np.array(global_labels), cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
#    print("AvgPrec:"," Prec:",cv_results.mean()," Recall:",cv_results.mean())
    
    model = model.fit(trainDataGlobal,trainLabelsGlobal)
    predict = model.predict(testDataGlobal)
    count = 0
    
    for idx, pred in enumerate(predict):
        if pred == testLabelsGlobal[idx]:
            count+=1
    acc = 1.0*count/len(predict)
    print(name, "Accuracy :", str(acc))
    
#    predict = model.predict(trainDataGlobal)
#    count = 0
    
#    for idx, pred in enumerate(predict):
#        if pred == trainLabelsGlobal[idx]:
#            count+=1
#    acc = 1.0*count/len(predict)
#    
    
#    print(name,"testAcc: ",str(acc))
    
    y_pred = cross_val_predict(model,testDataGlobal,testLabelsGlobal,cv=kfold)
    conf_mat = confusion_matrix(testLabelsGlobal,y_pred)
    print(conf_mat)
    print(classification_report(testLabelsGlobal,y_pred))
    

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

# to visualize results
import matplotlib.pyplot as plt

# create the model - Random Forests
#clf  = RandomForestClassifier(n_estimators=100, random_state=9)
clf = LogisticRegression(random_state=9)

# fit the training data to the model
#clf.fit(trainDataGlobal, trainLabelsGlobal)
clf.fit(np.array(global_features), np.array(global_labels))
# path to test data
test_path = "dataset/test"
my_test_path = 'dataset/global_data/test'
# loop through the test images
"""
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
"""
#attempt at confusion matrix
ctr = 0
for idx, sample in enumerate(testDataGlobal):
    # read the image
#    image = cv2.imread(file)

    # resize the image
#    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
#    fv_hu_moments = fd_hu_moments(image)
#    fv_haralick   = fd_haralick(image)
#    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
#    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # predict label of test image
#    prediction = clf.predict(global_feature.reshape(1,-1))[0]

    # show predicted label on image
#    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
#    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#    plt.show()
    prediction = clf.predict(sample.reshape(1,-1))[0]
    if prediction == testLabelsGlobal[idx]:
        ctr+=1
print("ACC: ", 1.0*ctr/testDataGlobal.shape[0])
    
    

"""
test_yy=[]
test_yy_pred = []

for idx,folder in enumerate(train_labels):
#    print(type(folder))
    for file in glob.glob(my_test_path+"/"+folder + "/*.jpg"):
        # read the image
        image = cv2.imread(file)
    
        # resize the image
        image = cv2.resize(image, fixed_size)
    
        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
    
        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    
        # predict label of test image
        prediction = clf.predict(global_feature.reshape(1,-1))[0]
    
        # show predicted label on image
#        cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    
        # display the output image
#        plt.imshow(cv2.cv1tColor(image, cv2.COLOR_BGR2RGB))
#        plt.show()
        
#        print("Actual: ", folder," | Predicted: ",train_labels[prediction], "  |loss: ",train_labels.index(folder),prediction)
        test_yy.append(train_labels.index(folder))
        test_yy_pred.append(prediction)
        
        
        
#    break
test_yy = np.array(test_yy)
test_yy_pred = np.array(test_yy_pred)

#for idx, pred in enumerate(test_yy_pred):
#    print("yy: ",test_yy[idx],"  |   yy_hat: ",pred)
#print(test_yy.shape,test_yy_pred.shape)

np.save("NB_test_labels",test_yy)
np.save("NB_test_pred",test_yy_pred)
np.save("NB_labelnames",np.array(train_labels))
ctr = 0
for idx,pred in enumerate(test_yy_pred):
    print(pred,test_yy[idx],)
    print(pred==test_yy[idx])
#    print(type(pred),type(test_yy[idx]))
    if pred == test_yy[idx]:
        
        ctr+=1
print("Accuracy:",1.0*ctr/test_yy.shape[0])
"""
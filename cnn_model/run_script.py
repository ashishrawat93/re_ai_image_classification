import subprocess
import os
import numpy as np
# python -m scripts.label_image     --graph=tf_files/retrained_graph.pb      --image=image_14.jpg

w_dir='/Users/ashishrawat/Desktop/tensorflow_local'

test_data = '/Users/ashishrawat/Desktop/tensorflow_local/test/'

train_labels = os.listdir(test_data)
train_labels.sort()
print(train_labels)
train_labels = train_labels[1:]
images_per_class=24

tf_test_y = []
tf_test_y_pred = []

g_cnt = 0

for idx,folder in enumerate(train_labels):
    print("folder:",folder)
    dir = os.path.join(test_data, folder)
    for x in range(1, images_per_class + 1):
        print("image_"+str(x)+" : ",end=" ")
        img = dir + "/image_"+str(x)+".jpg"
        p = subprocess.Popen('python -m scripts.label_image     --graph=tf_files/retrained_graph.pb      --image='+img, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        cnt=0
        y_hat=None
        for line in p.stdout.readlines():
            if cnt ==6:
                str_line = line.decode("utf-8")
                ls = str_line.split(" ")
                print(ls[0])
                y_hat = str(ls[0])
            # print(cnt, " : ", line)
            cnt+=1
        print( "y_hat: ",y_hat,"  ",end=" ")
        tf_test_y.append(train_labels.index(folder))
        tf_test_y_pred.append(train_labels.index(y_hat))

        print(tf_test_y[g_cnt]," | ",tf_test_y_pred[g_cnt])
        g_cnt+=1


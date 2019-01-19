'''
Created on 01 Jan 2019

@author: maree
'''
from ai.dnn import NeuralNetClass
from preprocess_mnist import DataWrapperClass
from pathlib import Path
import numpy as np
import  pandas as pd
import pickle
import os
cwd = os.getcwd()
data_c = DataWrapperClass(cwd)

def df_print(df, round_num=6):
    print(np.round(df, round_num))
    
    
'''
Helper functions to analyze performance analyzinf cunfusion matrix of DNN
'''
def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements        

'''
test for pickle serialized data, and create one if not available
'''
file = cwd + '/data/' + data_c.get_name()

pickled_file = Path( file )
if not pickled_file.is_file():
    print('Pickled MNIST data does not exists. Generate new pickled data set.')
    data_c.gen_file( )

'''
load pickled data set and extraxt data for processing
'''
with open( file , "br") as fh:
    data = pickle.load(fh)
train_imgs             = data[0]
test_imgs              = data[1]
train_labels           = data[2]
test_labels            = data[3]
train_labels_one_hot   = data[4]
test_labels_one_hot    = data[5]
image_size             = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels           = image_size * image_size

'''
Deep learning
'''
dnn_c = NeuralNetClass([image_pixels,100,no_of_different_labels],0.1)
confusion_matrix = np.zeros([no_of_different_labels,no_of_different_labels])


'''
Run experiment
'''
X = train_imgs
Y = train_labels_one_hot

dnn_c.train(X, Y);
dnn_c.train(X, Y);
dnn_c.train(X, Y);

X = test_imgs
Y = test_labels_one_hot

dnn_c.predict(X, Y)

'''
Analyze performance of DNN network on one Epoch 
'''
cm = dnn_c.cm()

print("label precision recall")
for label in range(10):
    print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")
    
print("precision total:", precision_macro_average(cm))
print("recall total:", recall_macro_average(cm))
print("accuracy:",accuracy(cm))


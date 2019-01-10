'''
Created on 01 Jan 2019

@author: maree
'''
from machine_learning import DeepNetworkClass 
import  numpy as np
import  pandas as pd
import random
'''
Binary clasification data set
'''
def generate_binary_dataset(random_seed=1104):
    random.seed(random_seed)
    randX = random.sample(range(1,9), 8)
    mapping_dict = {1:[0,0,0],
                 2:[0,0,1],
                 3:[0,1,0],
                 4:[0,1,1],
                 5:[1,0,0],
                 6:[1,0,1],
                 7:[1,1,0],
                 8:[1,1,1]}
    X = []
    for r in randX:
        X.append(mapping_dict[r])

    y = []
    randy = random.sample(range(1,9), 8)
    for el in randy:
        if el % 2 == 0:
            y.append([0])
        else:
            y.append([1])


    X = np.array(X)
    y = np.array(y)
    return X, y
'''
Non-linear data classification set
'''
def generate_nonlinear_dataset(feature_size = 3, datapoints_size = 10 ):
    for i in range(0,datapoints_size):
        if i==0:
            X,Y = sample_fn( feature_size );
        else:
            x,y = sample_fn( feature_size );
            X = np.vstack([X, x])
            Y = np.vstack([Y, y])
    
    return (X,Y)

def sample_fn( feature_size ):
    x = np.random.rand(feature_size );
    for i in range( feature_size ):
        y = x[i]**(i+1) ;
    return (x,y)

def df_print(df, round_num=6):
    print(np.round(df, round_num))

'''
Generate data points and setup NN
'''
X,Y = generate_binary_dataset()

dnn_c = DeepNetworkClass(3,4,5,1)
dnn_c.train(X, Y)

'''
Run experiment
'''
losses = []
epochs=1000;
epochs_list = []
for i in range(epochs+1):
    dnn_c.train(X, Y);
    if i % (epochs / 10) == 0:
        prediction,err,loss_epoch = dnn_c.predict(X, Y);
        epochs_list.append(i)
        losses.append(loss_epoch)

df_print( pd.DataFrame({'epoch' : epochs_list,'loss' : losses}) );
df_print( pd.DataFrame(data=np.concatenate([prediction[None].T, Y, err[None].T], axis=1),columns=['Prediction', 'Actual','Error']) )
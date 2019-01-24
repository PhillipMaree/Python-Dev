'''
Created on 01 Jan 2019

@author: maree
'''
from dnn_bSGD import NeuralNetClass 
import  numpy as np
import  pandas as pd
import random
import time
'''
Test function 
'''
def test_dataset(nx,ny,ns):
    return (np.random.rand(ns,nx),np.random.rand(ns,ny))

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
def generate_nonlinear_dataset(feature_size = 3, output_size = 1, datapoints_size = 10 ):
    for i in range(0,datapoints_size):
        if i==0:
            X,Y = sample_fn( feature_size, output_size );
        else:
            x,y = sample_fn( feature_size, output_size );
            X = np.vstack([X, x])
            Y = np.vstack([Y, y])
    
    return (X,Y)

def sample_fn( feature_size, output_size ):
    x = np.random.rand( feature_size )
    y = np.zeros(output_size)
    
    for i in range( feature_size ):
        for j in range( output_size ):
            y[j] += x[i]**j + x[i]**x[i] 
    return (x,y)

def df_print(df, round_num=6):
    print(np.round(df, round_num))
    
'''
Random sample a mini-batch size m from a training set 
'''  
def sample_batch(X,Y,batch_size ):
    random_idx = random.sample(range( np.size(X, 0) ),  batch_size )
        
    x_batch = np.zeros([np.size(X, 1),batch_size])
    y_batch = np.zeros([np.size(Y, 1),batch_size])    
    for i in range( len(random_idx) ):        
        x_batch[:,i] = X[random_idx[i]]
        y_batch[:,i] = Y[random_idx[i]]  
    return (x_batch, y_batch)  
     
'''
Generate data points and setup NN
'''
X_training,Y_training = generate_binary_dataset();
X_testing,Y_testing = generate_binary_dataset()

learning_rate = 0.2
num_feature = np.size(X_training,1)
num_output = np.size(Y_training,1)

dnn_c = NeuralNetClass([num_feature,4,num_output],2)

'''
Run experiment
'''
itr = 10000;
batch_size = 8
num_examples = np.size(X_training,0)
try:
    assert num_examples%batch_size == 0
except AssertionError as error:
    print('Batch size needs to be a factor of number of examples')

t0 = 0;
dt = 0;
for i in range(itr):
    x_batch, y_batch = sample_batch(X_training,Y_training,batch_size )
    t0 = time.time()
    #print('Training')
    dnn_c.train( x_batch, y_batch )      # mini-batch generate
    dt += time.time() - t0
    if i*batch_size % num_examples == 0: # epoch counter
        epoch_i = i*batch_size/num_examples
        if epoch_i % 500 == 0:
            #print('Prediction')
            
            prediction,loss,mean_loss = dnn_c.predict(X_testing.T, Y_testing.T)
            print('Epoch %3d with loss %.5f with T %.5f secs' % (epoch_i,mean_loss,dt))            
            dt = 0;
        
df_print( pd.DataFrame(data=np.concatenate([prediction.T, Y_testing, loss.T], axis=1),columns=['Prediction', 'Actual','Loss']) )


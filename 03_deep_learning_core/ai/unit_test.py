'''
Created on 01 Jan 2019

@author: maree
'''
from dnn_bSGD import NeuralNetClass as dnn_bSGD 
from dnn_SGD import NeuralNetClass as dnn_SGD
import  numpy as np
import  pandas as pd
import random
import time

def df_print(df, round_num=6):
    print(np.round(df, round_num))

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
Random sample a mini-batch size m from a training set 
'''  
def sample_batch(X,Y,batch_size=1 ):
    random_idx = random.sample(range( np.size(X, 0) ),  batch_size )
        
    x_batch = np.zeros([np.size(X, 1),batch_size])
    y_batch = np.zeros([np.size(Y, 1),batch_size])    
    for i in range( len(random_idx) ):        
        x_batch[:,i] = X[random_idx[i]]
        y_batch[:,i] = Y[random_idx[i]]  
    return (x_batch, y_batch)  

'''
Unit tests based on binary example
'''
def test_dnn_bSGD( X_training,Y_training, X_testing,Y_testing, N, learning_rate, arch, batch_size = 8 ):

    dnn_c = dnn_bSGD(arch,learning_rate)
    
    num_examples = np.size(X_training,0)
    try:
        assert num_examples%batch_size == 0
    except AssertionError:
        print('Batch size needs to be a factor of number of examples')
    
    t0 = 0;
    dt = 0;
    for i in range( N ):
        x_batch, y_batch = sample_batch(X_training,Y_training,batch_size )
        t0 = time.time()
        #print('Training')
        dnn_c.train( x_batch, y_batch )      # mini-batch generate
        dt += time.time() - t0
        if i*batch_size % num_examples == 0: # epoch counter
            epoch_i = i*batch_size/num_examples
            if epoch_i % 100 == 0:
                
                prediction,loss,mean_loss = dnn_c.predict(X_testing.T, Y_testing.T)
                print('Epoch %3d with loss %.5f with T %.5f secs' % (epoch_i,mean_loss,dt))            
                dt = 0;
            
    df_print( pd.DataFrame(data=np.concatenate([prediction.T, Y_testing, loss.T], axis=1),columns=['Prediction', 'Actual','Loss']) )                

def test_dnn_SGD( X_training,Y_training, X_testing,Y_testing, N, learning_rate, arch ):

    dnn_c = dnn_SGD(arch,learning_rate)
    
    num_examples = np.size(X_training,0)
    
    t0 = 0;
    dt = 0;
    for i in range( N ):
        x_batch, y_batch = sample_batch(X_training,Y_training )
        t0 = time.time()
        #print('Training')
        dnn_c.train( x_batch, y_batch )      # mini-batch generate
        dt += time.time() - t0
        if i % num_examples == 0: # epoch counter
            epoch_i = i/num_examples
            if epoch_i % 100 == 0:
                
                prediction,loss,mean_loss = dnn_c.predict(X_testing, Y_testing)
                print('Epoch %3d with loss %.5f with T %.5f secs' % (epoch_i,mean_loss,dt))            
                dt = 0;
            
    df_print( pd.DataFrame(data=np.concatenate([prediction, Y_testing, loss[None].T], axis=1),columns=['Prediction', 'Actual','Loss']) ) 
    
'''
Run unit test
'''

#Data definition    
X_training,Y_training = generate_binary_dataset()
X_testing,Y_testing = generate_binary_dataset()

#hyper parameters
learning_rate = 1
N = 10000
arch = [np.size(X_training, axis=1),4,np.size(Y_training, axis=1)]

test_dnn_bSGD( X_training,Y_training, X_testing,Y_testing, N, learning_rate, arch )
test_dnn_SGD( X_training,Y_training, X_testing,Y_testing, N, learning_rate, arch )
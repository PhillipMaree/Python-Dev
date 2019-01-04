'''
Created on 01 Jan 2019

@author: maree
'''
from machine_learning import DeepNetworkClass 
import  numpy as np
import  pandas as pd
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
X,Y = generate_nonlinear_dataset();

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
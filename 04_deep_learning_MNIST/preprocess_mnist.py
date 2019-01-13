'''
Created on 07 Jan 2019

@author: maree
'''
import numpy as np
import pickle


class DataWrapperClass:

    def __init__( self, cwd ):
        self.cwd = cwd
        self.filename = 'pickled_mnist.pkl'
    
    def gen_file( self ):
        cwd  = self.cwd
        '''
        Importing MNIST data set for image classification
        '''
        no_of_different_labels = 10             #  i.e. 0, 1, 2, 3, ..., 9
        data_path              = cwd+"/data/"
        print('1. Load MNIST data...')
        train_data = np.loadtxt(data_path + "mnist_train.csv",delimiter=",")
        test_data = np.loadtxt(data_path + "mnist_test.csv",delimiter=",") 
        
        
        # Scaling of data between data in (0,1)
        print('2. Scale MNIST data...')
        fac = 255  *0.99 + 0.01
        train_imgs = np.asfarray(train_data[:, 1:]) / fac
        test_imgs = np.asfarray(test_data[:, 1:]) / fac
        train_labels = np.asfarray(train_data[:, :1])
        test_labels = np.asfarray(test_data[:, :1])
        
        
        # Convert labels into one hot representation
        print('3. Hot label generation...')
        lr = np.arange(no_of_different_labels)
        train_labels_one_hot = (lr==train_labels).astype(np.float)
        test_labels_one_hot = (lr==test_labels).astype(np.float)
        
        # Condition data for excluding 0 and 1
        print('4. Sparse 0,1 from data...')
        train_labels_one_hot[train_labels_one_hot==0] = 0.01
        train_labels_one_hot[train_labels_one_hot==1] = 0.99
        test_labels_one_hot[test_labels_one_hot==0] = 0.01
        test_labels_one_hot[test_labels_one_hot==1] = 0.99
        
        # binary protocols for serializing and de-serializing a Python object structure
        print('5. Pickle Python object...')
        pickle_stream = cwd+"/data/"+self.filename
        with open( pickle_stream, "bw") as fh:
            data = (train_imgs, 
                    test_imgs, 
                    train_labels,
                    test_labels,
                    train_labels_one_hot,
                    test_labels_one_hot)
            pickle.dump(data, fh)
        
        print('Completed')
    
    def get_name( self ):
        return self.filename
    
    
    
    

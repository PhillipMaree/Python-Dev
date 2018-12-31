import numpy as np

class DeepNeuralNetClass:
    
    # Initializer / Instance Attributes
    def __init__( self , feature_size):
        print("Deep Neural network instantiated");
        
        self.n = feature_size; # number of input features
        self.m = 4; # number of neurons in a hidden layer 
        self.V = np.random.rand(self.m,self.n); # weights for inputs to first hidden layer of neurons
        self.W = np.random.rand(self.m,1); # weights for outputs to first hidden layer of neurons
                       
    '''
    Train the neural net for given block of input features X and measured outputs Y 
    '''
    def train( self, X, Y ):
        for row in range( 0,np.size(X,0) ):
            
            '''
            Features and meausured output
            '''
            x = X[row,:,None];
            y = Y[row,None]; 
            
            ''' 
            Forward propagation 
            '''
            a = self.A(x);
            b = self.B(a);
            c = self.C(b);
            p = self.P(c);
            
            '''
            Backwards propagation evaluating the sensitivities
            '''
            dAdV = self.A(x, False);
            dBdA = self.B(a, False);
            dCdW, dCdB = self.C(b, False);
            dPdC = self.P(c, False);
            dLdP = self.L(y, p, False);
            
            '''
            Chain rule to calculate the sensitivities of the weights
            '''
            dLdW = dLdP*dPdC*dCdW;
            dLdV = dLdP*dPdC*dCdB*dBdA*dAdV;
            
            '''
            Update the weights on the gradient search method
            '''
            self.W -= dLdW;
            self.V -= dLdV;
            
        pass
        
    '''
    For a new sample of features x, predict the output given the internal model structure
    '''
    def predict( self, X, Y ):
        ''' 
        Forward propagation 
        '''
        p = self.P(self.C(self.B(self.A(X.T))));
        m = np.mean((p - Y) ** 2) ;
        
        return (p,m);
    
    '''
    The activation function. Forward=false return the derivative dsigmoiddx
    '''
    def sigmoid(self, x, fward=True):
        if fward==True:
            return 1/(1+np.exp(-x));
        else:
            return self.sigmoid(x)*(1-self.sigmoid(x));
     
    
    '''
    This is the first evaluated neuron values of weighted features. Forward=false return the derivative dAdV
    '''
    def A(self, x, fward=True):
        if fward==True:
            return np.dot( self.V, x  ); 
        else:
            #return np.tile(x.T,[self.m,1]);
            return x.T;
        
    '''
    Calculate the active function for the current hidden layer. Forward=false return the derivative dBdA
    '''
    def B(self, a, fward=True):
        if fward==True:
            return self.sigmoid( a );
        else:
            return self.sigmoid( a, False ) ;
        
    '''
    Calculate the weighted output form hidden layer. Forward=false return the derivative dCdb dCdW
    '''
    def C(self, b, fward=True):    
        if fward==True:
            return b.T.dot(self.W);
        else:
            return (b, self.W);
        
    '''
    The cost activation function to measure performance. Forward=false return the derivative dPdc
    '''    
    def P(self,c, fward=True):
        if fward==True:
            return self.sigmoid( c );
        else:
            return self.sigmoid( c, False );
           
    '''
    The cost function to measure performance. Forward=false return the derivative dLdP
    '''   
    def L(self, y, P, fward=True):
        if fward==True:
            return 0.5*(y-P)**2;
        else:
            return -(y-P);
        
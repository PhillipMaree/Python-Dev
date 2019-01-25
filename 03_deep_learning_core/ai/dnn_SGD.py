import numpy as np
from pyllist import dllist

class ActivationFnClass():
    def __init__( self,neurons, fn_type='sigmoid' ):
        self.fn_type = fn_type
        self.dsigmadz = np.zeros([neurons])
    
    def sigma(self, *args ):       
        if len(args) ==1:
            z = args[0]
            # sigmoid activation function
            if self.fn_type == 'sigmoid':
                sigma = 1/(1+np.exp(-z))
                self.dsigmadz = sigma*(1-sigma)
            return sigma
        else:
            return self.dsigmadz

class NeuralLayerClass( ActivationFnClass  ):
    def __init__( self, layer_i, neurons_prev, neurons, learning_rate  ):
        ActivationFnClass.__init__(self, neurons )
        self.init(layer_i, neurons_prev, neurons )
        self.learning_rate=learning_rate
        
    # initialize layers weights and structure    
    def init(self,layer_i, neurons_prev, neurons  ): 
        print( self.info( layer_i, neurons_prev, neurons ) )
        
        self.x    = np.zeros([neurons_prev,1])           # x inputs to layers
        self.z    = np.zeros([neurons])                  # z = w*x + b
        self.c    = np.zeros([neurons])                  # c = sigmoid( z )
        self.w    = np.random.uniform(-1/np.sqrt(neurons_prev),1/np.sqrt(neurons_prev),[neurons,neurons_prev]    ) # weight combination of inputs x
        self.dldw = np.zeros([neurons,neurons_prev])     # cost sensitivity with change in weights
        self.b    = np.zeros([neurons,1])
        self.dldb = np.zeros([neurons,1])                # cost of sensitivity with change in bias

    def forward(self, x,update=True):
        if update==True:
            # update of weights
            self.w -= self.learning_rate*self.dldw
            self.b -= self.learning_rate*self.dldb
        # forward propagation given updated weights
        self.x = x
        self.z = np.dot(self.w,self.x) + self.b
        self.c = self.sigma( self.z )

        return self.c
       
    def backward(self, y):
        
        dldc = y
        dcdz = self.sigma()
        dldz = dldc*dcdz
        
        dzdw = np.tile(self.x.T, [np.size(self.w,0),1])
        dzdb = np.ones( self.b.shape )
        dzdx = self.w
        
        self.dldw = dldz*dzdw
        self.dldb = dldz*dzdb
        
        return np.dot(dldz.T,dzdx).T
    
    def info( self,  layer_i, neurons_prev, neurons  ):
        return "Add hidden layer %s with %d input(s) and %d neuron(s)" % ( layer_i, neurons_prev, neurons  )    

    
'''
Generic Deep learning architecture
'''
class NeuralNetClass:
    
        
    '''
    Variable dnn structure arguments stipulating minimum 3 layers (input,hidden,output). Arguments in are
    #neuron input layer, #neuron hidden layer 1, ...,#neuron hidden layer l, #neuron output layer L
    '''
    def __init__( self,layers, learning_rate=1 ): 
        
        MINIMUM_LAYER_COUNT = 3
        if(len(layers) < MINIMUM_LAYER_COUNT ):
            raise ValueError("Minimum of %d layers required in DNN" % MINIMUM_LAYER_COUNT )
        else:
            print("Initialize DNN with %d inputs, %d hidden layers with %d outputs]" % (layers[0],len(layers)-1,layers[-1]) )
            
            # construct dnn structure
            self.dnn_l = dllist();
            for i in range(1,len(layers)):
                self.dnn_l.append( NeuralLayerClass( i, layers[i-1],layers[i], learning_rate ))
                
        self.confusion_matrix = np.zeros([layers[-1],layers[-1]])
        
            
    def train(self, x_batch, y_batch):

        # forward propagation
        for layer_i in range(self.dnn_l.size): 
            x_batch = self.dnn_l[layer_i].forward(x_batch)
          
                  
        # evaluate cost
        y_batch = self.cost(y_batch, x_batch, False)
                     
        # backward propagation
        for layer_i in range(self.dnn_l.size):
            y_batch = self.dnn_l[self.dnn_l.size - layer_i-1].backward(y_batch)
    
    '''
    For a new sample of features x, predict the output given the internal model structure
    '''
    def predict( self, X, Y ):
        
        p = []
        l =[]
        for row in range( 0,np.size(X,0) ):
            
            '''
            Features and meausured output
            '''
            x = X[row,:,None]
            y = Y[row,None]
            
            # forward propagation
            for layer_i in range(self.dnn_l.size): 
                x = self.dnn_l[layer_i].forward(x,False)
            c_i = np.argmax(x)
            c_j = np.argmax(y)
            self.confusion_matrix[c_i,c_j]+=1
            
            if np.size(p)==0:
                p=x.T
            else:
                p = np.append(p,x.T,axis=0)
                                
            l=np.append(l,np.abs(x - y) )
        m = np.mean(l)    
  
        return (p,l,m);
 
    # cost function evaluated on output layer   
    def cost(self, y, P, fward=True):
        if fward==True:
            return 0.5*(y-P)**2
        else:
            return -(y-P) 
        
    def cm(self):
        return self.confusion_matrix                  


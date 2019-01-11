import numpy as np
from pyllist import dllist

class NeuralLayerClass( ):
    def __init__( self, layer_i, neurons_prev, neurons  ):
        self.init(layer_i, neurons_prev, neurons )
     
    # initialize layers weights and structure    
    def init(self,layer_i, neurons_prev, neurons  ): 
        print( self.info( layer_i, neurons_prev, neurons ) )
        
        self.x    = np.zeros([neurons_prev,1])           # x inputs to layers
        self.z    = np.zeros([neurons])                  # z = w*x + b
        self.c    = np.zeros([neurons])                  # c = sigmoid( z )
        self.w    = np.random.rand(neurons,neurons_prev) # weight combination of inputs x
        self.dldw = np.zeros([neurons,neurons_prev])     # cost sensitivity with change in weights
        self.b    = np.random.rand(neurons,1)            # bias weight
        self.dldb = np.zeros([neurons,1])                # cost of sensitivity with change in bias
                
    # neuron activation function
    def sigmoid(self, x,fward=True):
        if fward==True:
            return 1/(1+np.exp(-x))
        else:
            return self.sigmoid(x)*(1-self.sigmoid(x)) 
        
    def forward(self, x,update=True):
        if update==True:
            # update of weights
            self.w -= self.dldw
            self.b -= self.dldb
        # forward propagation given updated weights
        self.x = x
        self.z = np.dot(self.w,self.x) + self.b
        self.c = self.sigmoid( self.z )
        return self.c
       
    def backward(self, y):
        
        dldc = y
        dcdz = self.sigmoid( self.z, False )
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
    
        
    # Constructor
    def __init__( self, *args ):
        
        MINIMUM_LAYER_COUNT = 3
        if(len(args) < MINIMUM_LAYER_COUNT ):
            raise ValueError("Minimum of %d layers required in DNN" % MINIMUM_LAYER_COUNT )
        else:
            print("Initialize DNN with %d inputs, %d hidden layers with %d outputs]" % (args[0],len(args)-1,args[-1]) )
            
            # construct dnn structure
            self.dnn_l = dllist();
            for i in range(1,len(args)):
                self.dnn_l.append( NeuralLayerClass( i, args[i-1],args[i] ))
            
    def train(self, X, Y):
        for row in range( 0,np.size(X,0) ):
            
            '''
            Features and meausured output
            '''
            x = X[row,:,None]
            y = Y[row,:,None]
            
            # forward propagation
            for layer_i in range(self.dnn_l.size): 
                x = self.dnn_l[layer_i].forward(x)

                
            # evaluate cost
            l = self.cost(y, x)
            y = self.cost(y, x, False)
                       
            # backward propagation
            for layer_i in range(self.dnn_l.size):
                y = self.dnn_l[self.dnn_l.size - layer_i-1].backward(y)
                
    '''
    For a new sample of features x, predict the output given the internal model structure
    '''
    def predict( self, X, Y ):
        
        p = []
        err=[]
        for row in range( 0,np.size(X,0) ):
            
            '''
            Features and meausured output
            '''
            x = X[row,:,None]
            y = Y[row,None]
            
            # forward propagation
            for layer_i in range(self.dnn_l.size): 
                x = self.dnn_l[layer_i].forward(x,False)
            p = np.append(p,x)
            err=np.append(err,(x - y) ** 2)
        m = np.mean(err)    
  
        return (p,err,m);
 
    # cost function evaluated on output layer   
    def cost(self, y, P, fward=True):
        if fward==True:
            return 0.5*(y-P)**2;
        else:
            return -(y-P);                        
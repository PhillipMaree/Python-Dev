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
            elif self.fn_type == 'relu':
                z[z<=0] = 0
                sigma = z
                z[z>0] = 1 
                self.dsigmadz=z               
            return sigma
        else:
            return self.dsigmadz

class NeuralLayerClass( ActivationFnClass  ):
    def __init__( self, layer_i, neurons_prev, neurons, learning_rate  ):
        ActivationFnClass.__init__(self, neurons,'sigmoid' )
        self.init(layer_i, neurons_prev, neurons )
        self.alpha = learning_rate
        
    # initialize layers weights and structure    
    def init(self,layer_i, neurons_prev, neurons  ): 
        print( self.info( layer_i, neurons_prev, neurons ) )
        
        # forward pass weights
        self.w = np.random.uniform(-1/np.sqrt(neurons_prev),1/np.sqrt(neurons_prev),[neurons,neurons_prev]    ) # weight combination of inputs x
        self.beta = np.zeros([neurons,1])
        self.gamma = np.ones([neurons,1])
        
        # backward pass weights
        self.dldw = np.zeros([neurons,neurons_prev])     # cost sensitivity with change in weights
        self.dldbeta = np.zeros([neurons,1])                # cost of sensitivity with change in bias
        self.dldgamma = np.zeros([neurons,1])

    def forward(self, x):
        
        self.x = x # x is raw input, [row,col] = [#features,#batch examples]        
        
        self.u = np.dot(self.w,x)
        self.mu = self.u.mean(1)[None].T # mean row wise 
        self.var = self.u.var(1)[None].T # variance row wise
        self.y = (self.u - self.mu)/( np.sqrt(self.var + 1e-8 )   )
        self.z = self.gamma*self.y + self.beta
        self.c = self.sigma( self.z )

        return self.c

    def backward(self, y):
        
        m = np.size(self.x,1)
        
        dldc = y
        dcdz = self.sigma()
        dzdy = self.gamma
        dldy = dldc*dcdz*dzdy
        dldvar = np.sum( dldy*-1/2*( self.u-self.mu ), axis=1)[None].T*(self.var+ 1e-8)**(-3/2)
        dldmu = np.sum( dldy*-1/np.sqrt(self.var + 1e-8 ), axis=1 )[None].T + dldvar*(-2/m)*np.sum( self.u-self.mu, axis=1 )[None].T
        dldu = dldmu*np.ones(np.shape(self.u))/m + dldy/np.sqrt(self.var + 1e-8 ) + dldvar*2/m*( self.u-self.mu )
        dldx = np.dot( dldu.T, self.w ).T
        dldw = np.dot(dldu,self.x.T)
        dldbeta = np.sum( dldy, axis=1)[None].T
        dldgamma = np.sum( dldy*self.y, axis=1)[None].T
        
        self.w -= self.alpha*dldw
        self.beta -= self.alpha*dldbeta
        self.gamma -= self.alpha*dldgamma
    
        return dldx
    
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
    def __init__( self,layers, learning_rate=1): 
        
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
        
            
    def train(self, x_batch, y_batch ):
         
                    
        # forward propagation
        for layer_i in range(self.dnn_l.size): 
            x_batch = self.dnn_l[layer_i].forward(x_batch)
                  
        # evaluate cost
        y_batch = self.cost(y_batch,x_batch, False)
               
        # backward propagation
        for layer_i in range(self.dnn_l.size):
            y_batch = self.dnn_l[self.dnn_l.size - layer_i-1].backward(y_batch)
  
    '''
    For a new sample of features x, predict the output given the internal model structure
    '''
    def predict( self, x_batch, y_batch ):
        
        # forward propagation
        for layer_i in range(self.dnn_l.size): 
            x_batch = self.dnn_l[layer_i].forward(x_batch)
        
        p=x_batch
        l = np.abs(y_batch-p)    
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
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
        self.alpha = learning_rate
        
    # initialize layers weights and structure    
    def init(self,layer_i, neurons_prev, neurons  ): 
        print( self.info( layer_i, neurons_prev, neurons ) )
        
        # forward pass weights
        self.w = np.random.uniform(-1/np.sqrt(neurons_prev),1/np.sqrt(neurons_prev),[neurons,neurons_prev]    ) # weight combination of inputs x
        self.beta = np.zeros([neurons,1])
        self.gamma = np.random.uniform(-1/np.sqrt(neurons_prev),1/np.sqrt(neurons_prev),[neurons,1]    )  
        
        # backward pass weights
        self.dldw = np.zeros([neurons,neurons_prev])     # cost sensitivity with change in weights
        self.dldbeta = np.zeros([neurons,1])                # cost of sensitivity with change in bias
        self.dldgamma = np.zeros([neurons,1])

    def forward(self, x,update=True):
        
        self.x = x # x is raw input, [row,col] = [#features,#batch examples]        
        self.u = np.dot(self.w,x) # intermediage weighted input on all members in batch 
        self.mu = self.u.mean(1)[None].T # mean row wise 
        self.var = self.u.var(1)[None].T # variance row wise
        self.y = (self.u - self.mu)/( np.sqrt(self.var + 1e-8 )   )
        self.z = self.gamma*self.y + self.beta
        self.c = self.sigma( self.z )
        print(self.beta)
        return self.c

    def backward(self, y):
        
        m = np.size(self.x,1)
 
        
                
        e = 1e-8
        f = self.u
        g = self.mu
        h = (self.var + e)**-0.5
        
        dfdu = np.ones(np.size(self.u,axis=0))[None].T
        dgdu = np.ones(np.shape(self.u))/m
        dhdu = -(self.var + e)**(-3/2)*(self.u-self.mu)/m
        dudx = self.w
        dudw = self.x
        
        dldc = y
        dcdz = self.sigma()
        dzdy = self.gamma
        dydu = (dfdu-dgdu)*h + (f-g)*dhdu
        
        self.dldw =np.dot(dldc*dcdz*dzdy*dydu,dudw.T)
        self.dldbeta = 1/m*np.sum( dldc*dcdz, axis=1)[None].T
        self.dldgamma = 1/m*np.sum( dldc*dcdz*self.y, axis=1)[None].T
        
        self.w-=self.dldw
        self.beta-=self.dldbeta
        self.gamma-=self.dldgamma
        
        return np.dot((dldc*dcdz*dzdy*dydu).T,dudx).T
    
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
        y = y_batch
        P = x_batch
        dldc = self.cost(P, y, False)
        y_batch = dldc
        
                   
        # backward propagation
        for layer_i in range(self.dnn_l.size):
            y_batch = self.dnn_l[self.dnn_l.size - layer_i-1].backward(y_batch)
  
    '''
    For a new sample of features x, predict the output given the internal model structure
    '''
    def predict( self, x_batch, y_batch ):
        
        p = []
        err=[]
        for row in range( 0,np.size(x_batch,0) ):
            
            '''
            Features and meausured output
            '''
            x = x_batch[row,:,None]
            y = y_batch[row,None]
            
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
                                
            err=np.append(err,(x - y) ** 2)
        m = np.mean(err)    
  
        return (p,err,m);
 
    # cost function evaluated on output layer   
    def cost(self, y, P, fward=True):
        if fward==True:
            return 0.5*(y-P)**2
        else:
            return -(y-P) 
        
    def cm(self):
        return self.confusion_matrix                  
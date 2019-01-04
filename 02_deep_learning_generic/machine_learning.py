import numpy as np
from pyllist import dllist

'''
Defines the neuron structure used within the hidden layers of a deep learning network.
'''
class NeuronClass:
    
    # Constructor
    def __init__( self ):
        pass
        
    # neuron activation function
    def sigmoid(self, x, fward=True):
        if fward==True:
            return 1/(1+np.exp(-x))
        else:
            return self.sigmoid(x)*(1-self.sigmoid(x))
        
'''
Defines the structure of different layers within a deep learning architecture  
''' 
class FlattenLayerClass( ):
    

    def __init__( self, nodes ):
        self.layer = "Flatten"
        self.nodes = nodes;
        
    def forward(self, x, update=True ):
        y = x.flatten('C');
        return y
    
    def nodes(self):
        return self.nodes;
    
    def info(self):
        return "%s layer with %d node(s)" % (self.layer, self.nodes) 
    
                         
class DenseLayerClass( NeuronClass ):
    

    def __init__( self, nodes, dnn_l_prev ):
        self.layer= "Hidden"
        self.nodes= nodes;
        self.xn   = dnn_l_prev[-1].nodes               
        self.x    = np.zeros([1,self.xn])
        self.w    = np.random.rand(self.nodes,self.xn)
        self.dldw = np.zeros([self.nodes,self.xn])
        self.a    = np.zeros([1,self.nodes])
        self.b    = np.zeros([1,self.nodes])
        
    def forward(self, x ,update=True):
        if update==True:
            self.w-= self.dldw
        self.x = x
        self.a = np.dot(self.w,self.x.T)
        self.b = self.sigmoid( self.a )
        return self.b
    
    def backward(self, y):
        dldb = y
        dbda = self.sigmoid( self.a, False )
        dadw = np.tile(self.x.T,[self.nodes,1])
        dadx = self.w
        self.dldw = (dldb*dbda).T*dadw
        return np.dot(dldb*dbda,dadx)
    
    def nodes(self):
        return self.nodes;
        
    def info(self):
        return "%s layer with %d node(s)" % (self.layer, self.nodes)
    
class OutputLayerClass( DenseLayerClass ):
    

    def __init__( self, nodes, dnn_l_prev ):
        DenseLayerClass.__init__(self, nodes, dnn_l_prev)
        self.layer = "Output"
           
    def backward(self, y):
        dldb = self.cost(y, self.b, False)
        dbda = self.sigmoid( self.a, False )
        dadw = np.tile(self.x.T,[self.nodes,1])
        dadx = self.w 
        self.dldw = (dldb*dbda).T*dadw
        return np.dot(dldb*dbda,dadx)
    
    def cost(self, y, P, fward=True):
        if fward==True:
            return 0.5*(y-P)**2;
        else:
            return -(y-P);
    
'''
Generic Deep learning architecture
'''
class DeepNetworkClass:
    
        
    # Constructor
    def __init__( self, *args ):
        
        MINIMUM_LAYER_COUNT = 3
        
        if(len(args) < MINIMUM_LAYER_COUNT ):
            raise ValueError("Minimum of %d layers required in DNN" % MINIMUM_LAYER_COUNT )
        else:
            print("Initialize DNN with [%d features,%d hidden layers,%d outputs]" % (args[0],len(args)-2,args[-1]) )
            
            # empty double linked list defining DNN
            self.dnn_l = dllist();
            self.build_dnn( self.dnn_l, args )
            
    def train(self, X, Y):
        for row in range( 0,np.size(X,0) ):
            
            '''
            Features and meausured output
            '''
            x = X[row,:,None]
            y = Y[row,None]
            
            # forward propagation
            for layer_i in range(self.dnn_l.size): 
                x = self.dnn_l[layer_i].forward(x)
                       
            # backward propagation
            for layer_i in range(self.dnn_l.size-1):
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
            
    def build_dnn(self, dnn_l, args ):
        
        dnn_l.append( FlattenLayerClass( args[0] ) );
        for i in range(1,len(args)-1):
            dnn_l.append( DenseLayerClass( args[i], dnn_l ) );
        dnn_l.append( OutputLayerClass( args[len(args)-1], dnn_l ) )

    
    
            
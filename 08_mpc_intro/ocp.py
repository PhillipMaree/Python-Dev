import casadi as ca
import numpy as np

''' This class defines the optimal control problem to be solved'''


class OCP:

    class Signal:
        def __init__(self,var_name,var_lb,var_ub,var_n):
            self.var_name = var_name
            self.lb = var_lb
            self.ub = var_ub
            self.n = var_n
            self.var = ca.MX.sym(var_name,1,self.n)

        def var(self): return self.var_name
        def lb(self): return self.lb
        def ub(self): return self.ub
        def s(self): return self.var

    def __init__(self,name="Empty OPC formulation",n_x=0,n_u=0):
        # initialize minimum characteristics
        self.n_x = n_x
        self.n_u = n_u
        self.name = name

        # construct system dimensions
        self.x = self.Signal('x',self.xmin(), self.xmax(),self.n_x)
        self.u = self.Signal('u',self.umin(), self.umax(),self.n_u)

        # obtain core context of opc problem
        self.p = self.get_parameters()
        self.ode = self.get_ode()
        self.l = self.get_l()

        # construct functional
        self.fn = ca.Function('fn', [self.x.s(), self.u.s()], [self.ode, self.l], ['x', 'u'], ['ode', 'l'])

    def get_name(self): return self.name
    def get_parameters(self): return []
    def get_ode(self): return ca.vertcat()
    def get_l(self): return 0
    def umin(self): return 0
    def umax(self): return 0
    def xmin(self): return 0
    def xmax(self): return 0


''' New OPC formulations should be defined here, inheriting the base OCP class '''


class CstrOCP(OCP):

    def __init__(self):
        self.n_x = 3
        self.n_u = 1
        OCP.__init__(self,"Acetylene hydrogenation process",self.n_x,self.n_u)

    def get_parameters(self):
        return {'sigma1': 1000, 'sigma2': 472, 'beta': 23}

    def get_ode(self):

        x1 = self.x.s()[0]
        x2 = self.x.s()[1]
        x3 = self.x.s()[2]

        u1 = self.u.s()[0]

        sigma1 = self.p['sigma1']
        sigma2 = self.p['sigma2']
        beta = self.p['beta']

        a = sigma1*x1*x2/(1+beta*x1)
        b = sigma2*x2**0.5*x3/(1+beta*x1)

        dx1 = 1-x1-a
        dx2 = u1-x2-a-b
        dx3 = -x3+a-b

        return ca.vertcat(dx1, dx2, dx3)

    def get_l(self):
        x3 = self.x.s()[2]
        return -x3

        # variable constraints
    def umin(self):
        return np.array([0.1])

    def umax(self):
        return np.array([5])

    def xmin(self):
        return np.array([0,0,0])

    def xmax(self):
        return np.array([1,1,1])




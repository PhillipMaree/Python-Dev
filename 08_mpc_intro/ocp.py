import casadi as ca
import numpy as np


class Var:
    def __init__(self,name,lb,ub,dim):
        self.var_name = name
        self.var_lb = lb
        self.var_ub = ub
        self.var_dim = dim
        self.mx_var = ca.MX.sym(name,1,self.var_dim)

    def name(self): return self.var_name
    def lb(self): return self.var_lb
    def ub(self): return self.var_ub
    def var(self): return self.mx_var
    def dim(self): return self.var_dim


# This class defines the optimal control problem to be solved according to the terminology of Betts, 2010 Ch4
class OCP:

    def __init__(self, n_y=0, n_u=0, name="Empty OPC formulation"):
        # problem name description
        self.name = name

        # time indexing
        self.t = Var('t', 0, np.inf, 1)
        # dynamic state variables
        self.y = Var('y',self.y_min(), self.y_max(),n_y)
        # dynamic control variables
        self.u = Var('u',self.u_min(), self.u_max(),n_u)
        # system parameters
        self.p = self.parameters()

        # dynamics of system in explicit ODE form
        self.f = self.f_fn()
        # algebraic path constraints
        self.g = self.g_fn()
        # cost functional
        self.w = self.w_fn()

        # casadi ocp functional
        self.F = self.F_fn()

    # casadi funcitonal to be called in NLP problem formulation
    def F_fn(self):
        if self.g.nnz() == 0:
            return ca.Function('fn_yu_fw', [self.t.var(), self.y.var(), self.u.var()], [self.f, self.w], ['t', 'y', 'u'], ['f', 'w'])
        else:
            return ca.Function('fn_yu_fgw', [self.t.var(), self.y.var(), self.u.var()], [self.f, self.g, self.w], ['t', 'y', 'u'], ['f', 'g', 'w'])

    # problem specific system parameters
    def parameters(self): return []
    # ocp continuous functions
    def f_fn(self): return ca.vertcat()
    def g_fn(self): return ca.vertcat()
    def w_fn(self): return ca.vertcat()
    # differential state equation bounds
    def y_min(self): return np.array([])
    def y_max(self): return np.array([])
    # piece-wise constant control bounds
    def u_min(self): return np.array([])
    def u_max(self): return np.array([])
    # algebraic path constraint bounds
    def g_min(self): return np.array([])
    def g_max(self): return np.array([])

    # helper functions
    def has_DAE(self):
        if self.F.name() == 'fn_yu_fgw':
            return True
        return False


''' New OPC formulations should be defined here, inheriting the base OCP class '''


class CstrOCP(OCP):

    def __init__(self):
        OCP.__init__(self, 3, 1, "Acetylene hydrogenation process")

    def parameters(self):
        return {'sigma1': 1000, 'sigma2': 472, 'beta': 23}

    def f_fn(self):

        x1 = self.y.var()[0]
        x2 = self.y.var()[1]
        x3 = self.y.var()[2]

        u1 = self.u.var()[0]

        sigma1 = self.p['sigma1']
        sigma2 = self.p['sigma2']
        beta = self.p['beta']

        a = sigma1*x1*x2/(1+beta*x1)
        b = sigma2*x2**0.5*x3/(1+beta*x1)

        odef1 = 1-x1-a
        odef2 = u1-x2-a-b
        odef3 = -x3+a-b

        return ca.vertcat(odef1, odef2, odef3)

    def w_fn(self):
        x3 = -self.y.var()[2]
        return x3

        # variable constraints
    def u_min(self):
        return np.array([1e-3])

    def u_max(self):
        return np.array([5.0])

    def y_min(self):
        return np.array([1e-3,1e-3,1e-3])

    def y_max(self):
        return np.array([1.0,1.0,1.0])






import casadi as ca
import numpy as np

# This class defines the optimal control problem to be solved according to the terminology of Betts, 2010 Ch4
class OCP:

    class Variable:
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

    def __init__(self,name="Empty OPC formulation",n_y=0, n_u=0):
        # problem name description
        self.name = name
        # dynamic state variables
        self.y = self.Variable('y',self.y_min(), self.y_max(),n_y)
        # dynamic control variables
        self.u = self.Variable('u',self.u_min(), self.u_max(),n_u)
        # system parameters
        self.p = self.parameters()
        # dynamics of system in explicit ODE form
        self.f = self.state_equations()
        # algebraic path constraints
        self.g = self.path_constraints()
        # cost functional
        self.w = self.quadrature_functions()
        # casadi ocp functional
        self.ocp_fn = self.casadi_functional()

        # casadi ocp functional with ODE and DAE functionals
        self.ocp_fn = ca.Function('ocp_fn', [self.y.var(), self.u.var()], [self.f, self.g, self.w], ['y', 'u'], ['f', 'g', 'w'])
        # casadi ocp functional with ODE functionals
        self.ocp_fn = ca.Function('ocp_fn', [self.y.var(), self.u.var()], [self.f, self.w], ['y', 'u'], ['f', 'w'])

    # casadi funcitonal to be called in NLP problem formulation
    def casadi_functional(self):
        if self.g.nnz() == 0:
            return ca.Function('fn_yu_fw', [self.y.var(), self.u.var()], [self.f, self.w], ['y', 'u'], ['f', 'w'])
        else:
            return ca.Function('fn_yu_fgw', [self.y.var(), self.u.var()], [self.f, self.g, self.w], ['y', 'u'], ['f', 'g', 'w'])
    # determine if
    def has_DAE(self):
        if self.ocp_fn.name() == 'fn_yu_fgw':
            return True
        return False

    # functions to be overwritten in problem child OCP class
    def parameters(self): return []
    def state_equations(self): return ca.vertcat()
    def path_constraints(self): return ca.vertcat()
    def quadrature_functions(self): return ca.vertcat()
    def y_min(self): return np.array([])
    def y_max(self): return np.array([])
    def y_min(self): return np.array([])
    def y_max(self): return np.array([])


''' New OPC formulations should be defined here, inheriting the base OCP class '''


class CstrOCP(OCP):

    def __init__(self):
        OCP.__init__(self,"Acetylene hydrogenation process", 3, 1)

    def parameters(self):
        return {'sigma1': 1000, 'sigma2': 472, 'beta': 23}

    def state_equations(self):

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

    def quadrature_functions(self):
        return -self.y.var()[2]

        # variable constraints
    def u_min(self):
        return np.array([1e-3])

    def u_max(self):
        return np.array([5.0])

    def y_min(self):
        return np.array([1e-3,1e-3,1e-3])

    def y_max(self):
        return np.array([1.0,1.0,1.0])




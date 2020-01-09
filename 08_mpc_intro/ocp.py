import casadi as ca
import numpy as np

''' This class defines the optimal control problem to be solved'''


class OCP:
    def __init__(self):

        # OCP formulation for the underlying problem
        self.problem = "Acetylene hydrogenation process"

        # system states
        x1 = ca.SX.sym('x1')
        x2 = ca.SX.sym('x2')
        x3 = ca.SX.sym('x3')

        # control inputs
        u1 = ca.SX.sym('u1')

        # problem constants
        sigma1 = 1000
        sigma2 = 472
        beta = 23
        a = sigma1*x1*x2/(1+beta*x1)
        b = sigma2*x2**0.5*x3/(1+beta*x1)

        # ode formulation
        dx1 = 1-x1-a
        dx2 = u1-x2-a-b
        dx3 = -x3+a-b

        self.opc = {}
        # concatenate variables
        self.opc['x'] = ca.vertcat(x1, x2, x3)
        self.opc['u'] = ca.vertcat(u1)
        self.opc['dx'] = ca.vertcat(dx1, dx2, dx3)

        # objectives
        self.opc['l'] = -x3

        # Continuous time dynamics
        self.opc['f'] = ca.Function('f', [self.opc['x'], self.opc['u']], [self.opc['dx'], self.opc['l']], ['x', 'u'], ['dx', 'l'])

        # bounds on states
        self.opc['x_min'] = np.array([0,0,0])
        self.opc['x_max'] = np.array([1,1,1])

        # bound on controls
        self.opc['u_min'] = np.array([0.1])
        self.opc['u_max'] = np.array([5])



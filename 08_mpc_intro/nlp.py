import casadi as ca
import numpy as np
import pandas as pd
from ocp import CstrOCP as ocp


class Extract:

    def __init__(self,y_n, u_n, N, K):
        self.y_n = y_n
        self.u_n = u_n
        self.N = N
        self.K = K

        # pandas frame to report solution
        label = ['t']
        for i in range(1,self.y_n+1):
            label.append('y_'+str(i))
        for i in range(1,self.u_n+1):
            label.append('u_'+str(i))
        self.df = pd.DataFrame(columns=['time', 'y1', 'y2', 'y3', 'u'])

    def sol(self,t,sol):
        # translate symbolic array to flatten np array
        sol = np.array(sol).flatten()
        # transpose the well conditioned for N-stages ([0,N-1]) excl N+1 stage
        sol_N = sol[0:self.N*((1+self.K)*self.y_n + self.u_n)].reshape(self.N,(1+self.K)*self.y_n + self.u_n)
        # append [t,y,u] to pandas data frame
        for i in range(self.N):
            # retain solution for phase i
            sol_i = sol_N[i]
            # extract state and collocation points and reshape [1+K,y_n]
            sol_y = sol_i[0:(1+self.K)*self.y_n].reshape(1+self.K,self.y_n)
            # extract controls and reshape [1,u_n]
            sol_u = sol_i[(1+self.K)*self.y_n-self.u_n+1:].reshape(1,self.u_n)
            # append pandas dataframe
            for j in range(1+self.K):
                i_j_idx = i*(1+self.K)+j
                self.df.loc[i_j_idx] = np.append(t[i_j_idx], np.append(sol_y[j], sol_u[0]))
        # append final state boundary at i=N (extend previous control for completeness)
        N_idx = self.N*(1+self.K)
        self.df.loc[N_idx] = np.append(t[N_idx], np.append(sol[-self.y_n:], sol_u[0]))

        return self.df


# construct the NLP from the inherited OCP problem. NLP formulation to the terminology of Betts, 2010, Ch4
class NLP:

    def __init__(self, nlp_config):

        # collocation specific coefficients
        self.B, self.C, self.D, self.K, self.tau = self.collocation(nlp_config['collocation_method'], nlp_config['collocation_degree'])

        # instantiate ocp structure
        self.ocp = ocp()

        # nlp specific parameters
        self.N = nlp_config['N']
        self.h = nlp_config['h']
        self.y_n = self.ocp.y.dim()
        self.u_n = self.ocp.u.dim()

        # for statistical purposes
        self.u_var_N = self.N*self.u_n
        self.y_var_N = (self.N+1)*self.y_n
        self.c_var_N = self.N*self.K*self.y_n
        self.nlp_var_N = self.u_var_N + self.y_var_N + self.c_var_N

        # fast index offsetting
        self.y_offset = 0
        self.c_offset = 1
        self.u_offset = (1 + self.K)
        self.phase_offset = (1 + self.K) + 1

        # nlp structure
        self.nlp_struct = {'t': [], 'J': 0, 'z': [], 'z0': [], 'lbz': [], 'ubz': [], 'g': [], 'lbg': [], 'ubg': [], 'solver': lambda z0, lbz, ubz, lbg, ubg: ()}

        # construct empty NLP variable structure
        for i in range(self.N):
            self.add_time((i+self.tau[0])*self.h)
            # add differential variable on phase boundary
            self.add_nlp_var(ca.MX.sym( self.ocp.y.name() + '_' + str(i), self.ocp.y.dim() ), self.ocp.y.lb(), self.ocp.y.ub(), self.ocp.y.lb())
            # add collocation variables
            for j in range(1,self.K+1):
                self.add_time((i+self.tau[j])*self.h)
                self.add_nlp_var(ca.MX.sym( self.ocp.y.name() + '_' + str(i) + '_' + str(j), self.ocp.y.dim() ), self.ocp.y.lb(), self.ocp.y.ub(), self.ocp.y.lb())
            # add control variable
            self.add_nlp_var(ca.MX.sym( self.ocp.u.name() + '_' + str(i), self.ocp.u.dim() ), self.ocp.u.lb(), self.ocp.u.ub(), self.ocp.u.lb())
        self.add_time(self.N*self.h)
        # add terminal differential variable
        self.add_nlp_var(ca.MX.sym( self.ocp.y.name() + '_' + str(self.N), self.ocp.y.dim() ), self.ocp.y.lb(), self.ocp.y.ub(), self.ocp.y.lb())

        # construct nlp problem
        self.init_nlp()

        # object to extract data
        self.extract = Extract(self.ocp.y.dim(), self.ocp.u.dim(), self.N, self.K)

    def add_time(self, t_i_j):
        self.nlp_struct['t'].append(t_i_j)

    def add_nlp_var(self, var, lb, ub, var0):
        self.nlp_struct['z'].append(var)
        self.nlp_struct['lbz'].append(lb)
        self.nlp_struct['ubz'].append(ub)
        self.nlp_struct['z0'].append(var0)

    def add_nlp_res(self, res, lbg, ubg):
        self.nlp_struct['g'].append(res)
        self.nlp_struct['lbg'].append(lbg)
        self.nlp_struct['ubg'].append(ubg)

    def add_cost(self, cost):
        self.nlp_struct['J'] += cost

    def init_solver(self):
        self.nlp_struct['z'] = ca.vertcat(*self.nlp_struct['z'])
        self.nlp_struct['g'] = ca.vertcat(*self.nlp_struct['g'])
        self.nlp_struct['z0'] = np.concatenate(self.nlp_struct['z0'])
        self.nlp_struct['lbz'] = np.concatenate(self.nlp_struct['lbz'])
        self.nlp_struct['ubz'] = np.concatenate(self.nlp_struct['ubz'])
        self.nlp_struct['lbg'] = np.concatenate(self.nlp_struct['lbg'])
        self.nlp_struct['ubg'] = np.concatenate(self.nlp_struct['ubg'])

        # construct nlp solver
        prob = {'f': self.nlp_struct['J'], 'x': self.nlp_struct['z'], 'g': self.nlp_struct['g']}
        self.nlp_struct['solver'] = ca.nlpsol('solver', 'ipopt', prob)

    def init_nlp(self):

        # lambda functions to gain offset access in nlp variable structure
        y = lambda k,obj=self: obj.nlp_struct['z'][k*self.phase_offset + self.y_offset ]
        c = lambda k,j,obj=self: obj.nlp_struct['z'][k*self.phase_offset + self.c_offset-1 + j]
        u = lambda k,obj=self: obj.nlp_struct['z'][k*self.phase_offset + self.u_offset]
        t = lambda i,j,obj=self: obj.nlp_struct['t'][i*(1+self.K)+j]

        for i in range(self.N):
            # continuity boundary constraints
            y_i_end = self.D[0]*y(i)
            for tau_j in range(1,self.K+1):
                y_i_end += self.D[tau_j]*c(i, tau_j)
            self.add_nlp_res( y_i_end - y(i+1), np.zeros(self.ocp.y.dim()), np.zeros(self.ocp.y.dim()))

            # residual constraints for collocation equations
            for tau_k in range(1, self.K+1):
                y_i_p = self.C[0, tau_k]*y(i)
                for tau_j in range(1, self.K+1):
                    y_i_p += self.C[tau_j, tau_k]*c(i,tau_j)
                #f_tau_k, w_tau_k = self.ocp.F(t(i,tau_j), c(i, tau_k), u(i)) ?
                f_tau_k, w_tau_k = self.ocp.F(t(i,tau_k), c(i, tau_k), u(i))
                self.add_nlp_res(self.h*f_tau_k - y_i_p, np.zeros(self.ocp.y.dim()), np.zeros(self.ocp.y.dim()))
                #self.add_cost(self.B[tau_k]*w_tau_k*self.h)

            # add phase cost
            for tau_j in range(self.K+1):
                f_tau_j, w_tau_j = self.ocp.F(t(i,tau_j), c(i,tau_j), u(i))
                self.add_cost(self.B[tau_j]*w_tau_j*self.h)

        # terminal cost
        f_tau_N, w_tau_N = self.ocp.F(t(self.N,0), y(self.N), u(i))
        self.add_cost( self.B[0]*w_tau_N*self.h )

        self.init_solver()

        # print user statistics on creation
        print("NLP consists of %d variables of which:\n"
              "\t%d differential variables\n"
              "\t%d piece-wise constant control variables\n"
              "\t%d collocation variables" % (self.nlp_var_N, self.y_var_N, self.u_var_N, self.c_var_N))

    def collocation(self, method, degree):

        if method == 'legendre':
            tau_root = np.append(ca.collocation_points(degree, method), 1)
        elif method == 'radau':
            tau_root = np.append(0, ca.collocation_points(degree, method))

        K = degree
        B = np.zeros(K + 1)
        C = np.zeros((K + 1, K + 1))
        D = np.zeros(K + 1)

        for j in range(K+1):
            # construct polynomial for j-th collocation point
            lj = np.poly1d([1])
            for k in range(K+1):
                if k != j:
                    lj *= np.poly1d( [1, -tau_root[k]] ) / ( tau_root[j] -tau_root[k] )
            # coefficients continuity equations
            D[j] = lj(1.0)
            # coefficients for collocation equations
            dlj = np.polyder(lj)
            for k in range(K+1):
                C[j, k] = dlj(tau_root[k])
            # coefficients for
            ilj = np.polyint(lj)
            B[j] = ilj(1.0)

        print("Eval Collocation points Int\n")
        print("tau root {}\n".format(tau_root))
        print("B {}\n".format(C))
        print("C {}\n".format(C))
        print("D {}\n".format(D))

        return B, C, D, K, tau_root

    def solve_ocp(self,t_k, *y_0):
        # enforce initial conditions
        self.nlp_struct['z0'][0:self.ocp.y.dim()] = y_0
        self.nlp_struct['lbz'][0:self.ocp.y.dim()] = y_0
        self.nlp_struct['ubz'][0:self.ocp.y.dim()] = y_0

        # call defined solver
        solution = self.nlp_struct['solver'](x0=self.nlp_struct['z0'], lbx=self.nlp_struct['lbz'], ubx=self.nlp_struct['ubz'], lbg=self.nlp_struct['lbg'],ubg=self.nlp_struct['ubg'])

        # extract data from optimal solution
        t_offset = t_k*self.h
        return self.extract.sol([t + t_offset for t in self.nlp_struct['t']], solution['x'])



import casadi as ca
import numpy as np
from ocp import CstrOCP

''' The defined OCP is transcribed into a NLP problem '''


class NLP(CstrOCP):
    def __init__(self, td, N):
        CstrOCP.__init__(self)

        self.nlp_config = {'collocation_method': 'legendre',
                           'collocation_degree': 3,
                           'time_horizon': td*N,
                           'num_phases': N,
                           'phase_step_size': td}
        # nlp structure
        self.nlp = {'f': 0, 'x': [], 'x0': [], 'lbx': [], 'ubx': [], 'g': [], 'lbg': [], 'ubg': [], 'solver': lambda x0, lbx, ubx, lbg, ubg : ()}

        # nlp plotting
        self.plot = {'x': [], 'u': []}

        # collocation polynomial coefficients
        self.B, self.C, self.D = self.collocation_polynomial()
        self.generate_nlp()

        print("NLP formulation for: %s" % self.get_name() )

    def collocation_polynomial(self):

        degree = self.nlp_config['collocation_degree']
        method = self.nlp_config['collocation_method']

        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(degree, method))

        # Coefficients of the quadrature function
        B = np.zeros(degree+1)
        # Coefficients of the collocation equation
        C = np.zeros((degree+1,degree+1))
        # Coefficients of the continuity equation
        D = np.zeros(degree+1)

        # Construct polynomial basis
        for j in range(degree+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(degree+1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(degree+1):
                C[j,r] = pder(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        return B, C, D

    def generate_nlp(self):

        # local NLP constants
        N = self.nlp_config['num_phases']
        h = self.nlp_config['phase_step_size']
        degree = self.nlp_config['collocation_degree']

        # "Lift" initial conditions
        x_k = ca.MX.sym('x_0', self.n_x)
        self.nlp['x'].append(x_k)
        self.nlp['lbx'].append(self.x.lb)
        self.nlp['ubx'].append(self.x.ub)
        self.nlp['x0'].append(self.x.lb)
        self.plot['x'].append(x_k)

        for phase_k in range(N):
            # piece-wise constant control at phase
            u_k = ca.MX.sym('u_' + str(phase_k))
            self.nlp['x'].append(u_k)
            self.nlp['lbx'].append(self.u.lb)
            self.nlp['ubx'].append(self.u.ub)
            self.nlp['x0'].append(1.1*self.u.lb)
            self.plot['u'].append(u_k)

            # add symbolic variables for phase k collocation points
            x_c = []
            for tau_j in range(degree):
                # boundary initial state condition (initial boundary condition)
                x_k_j = ca.MX.sym('x_'+str(phase_k)+'_'+str(tau_j),self.n_x)
                x_c.append(x_k_j)
                self.nlp['x'].append(x_k_j)
                self.nlp['lbx'].append(self.x.lb)
                self.nlp['ubx'].append(self.x.ub)
                self.nlp['x0'].append(1.1*self.x.lb)

            # assign equality constraints for polynomial interpolation
            x_k_f = self.D[0]*x_k
            for tau_j in range(degree):
                x_p = self.C[0, tau_j+1]*x_k
                for tau_r in range(degree): x_p += self.C[tau_r+1,tau_j+1]*x_c[tau_r]

                f_tau_j, q_tau_j = self.fn(x_c[tau_j],u_k)

                self.nlp['g'].append( h*f_tau_j - x_p ) # TODO
                self.nlp['lbg'].append(np.zeros(self.n_x))
                self.nlp['ubg'].append(np.zeros(self.n_x))

                x_k_f += self.D[tau_j+1]*x_c[tau_j]

                self.nlp['f'] += self.B[tau_j+1]*q_tau_j*h

            # create state variable for next phase
            x_k = ca.MX.sym('x_' + str(phase_k+1), self.n_x)
            self.nlp['x'].append(x_k)
            self.nlp['lbx'].append(self.x.lb)
            self.nlp['ubx'].append(self.x.ub)
            self.nlp['x0'].append(1.1*self.x.lb)
            self.plot['x'].append(x_k)

            # enforce continuity constraint
            self.nlp['g'].append( x_k_f - x_k ) # TODO
            self.nlp['lbg'].append(np.zeros(self.n_x))
            self.nlp['ubg'].append(np.zeros(self.n_x))

        # Concatenate vectors
        self.nlp['x'] = ca.vertcat(*self.nlp['x'])
        self.nlp['g'] = ca.vertcat(*self.nlp['g'])
        self.plot['x'] = ca.horzcat(*self.plot['x'])
        self.plot['u'] = ca.horzcat(*self.plot['u'])
        self.nlp['x0'] = np.concatenate(self.nlp['x0'])
        self.nlp['lbx'] = np.concatenate(self.nlp['lbx'])
        self.nlp['ubx'] = np.concatenate(self.nlp['ubx'])
        self.nlp['lbg'] = np.concatenate(self.nlp['lbg'])
        self.nlp['ubg'] = np.concatenate(self.nlp['ubg'])

        # construct nlp solver
        prob = {'f': self.nlp['f'], 'x': self.nlp['x'], 'g': self.nlp['g']}
        self.nlp['solver'] = ca.nlpsol('solver', 'ipopt', prob)

    def solve_nlp(self,*x_0):
        # enforce initial conditions
        self.nlp['x0'][0:self.n_x] = x_0[1:]
        self.nlp['lbx'][0:self.n_x] = x_0[1:]
        self.nlp['ubx'][0:self.n_x] = x_0[1:]
        #solve nlp problem
        return self.nlp['solver'](x0=self.nlp['x0'], lbx=self.nlp['lbx'], ubx=self.nlp['ubx'],lbg=self.nlp['lbg'],ubg=self.nlp['ubg'])

    def solve(self,*x_0):
        sol = self.solve_nlp(self,*x_0)
        trajectories = ca.Function('trajectories', [self.nlp['x']], [self.plot['x'], self.plot['u']], ['x'], ['x', 'u'])
        x_opt, u_opt = trajectories(sol['x'])
        x_opt = x_opt.full()
        u_opt = u_opt.full()
        return x_opt, u_opt

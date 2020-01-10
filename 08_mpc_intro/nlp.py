import casadi as ca
import numpy as np
from ocp import CstrOCP as ocp


# construct the NLP from the inherited OCP problem. NLP formulation to the terminology of Betts, 2010, Ch4
class NLP:


    def __init__(self, td, N):

        self.ocp = ocp()

        self.nlp_config = {'collocation_method': 'legendre',
                           'collocation_degree': 3,
                           'time_horizon': td * N,
                           'num_phases': N,
                           'phase_step_size': td,
                           'warm_start': True}

        # nlp structure
        self.nlp = {'J': 0, 'z': [], 'z0': [], 'lbz': [], 'ubz': [], 'g': [], 'lbg': [], 'ubg': [], 'warmstart': [], 'solver': lambda z0, lbz, ubz, lbg, ubg: ()}

        # nlp plotting
        self.plot = {'y': [], 'u': []}

        # collocation polynomial coefficients
        self.B, self.C, self.D = self.collocation_polynomial()

        # construct nlp problem
        self.generate_nlp()

        print("NLP formulation for: %s" % self.ocp.name)

    def collocation_polynomial(self):

        degree = self.nlp_config['collocation_degree']
        method = self.nlp_config['collocation_method']

        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(degree, method))
        # Coefficients of the quadrature function
        B = np.zeros(degree + 1)
        # Coefficients of the collocation equation
        C = np.zeros((degree + 1, degree + 1))
        # Coefficients of the continuity equation
        D = np.zeros(degree + 1)

        # Construct polynomial basis
        for j in range(degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(degree + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(degree + 1):
                C[j, r] = pder(tau_root[r])

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
        y_k = ca.MX.sym(self.ocp.y.name() + '_0', self.ocp.y.dim())
        self.nlp['z'].append(y_k)
        self.nlp['lbz'].append(self.ocp.y.lb())
        self.nlp['ubz'].append(self.ocp.y.lb())
        self.nlp['z0'].append(self.ocp.y.lb())
        self.plot['y'].append(y_k)

        for phase_k in range(N):
            # piece-wise constant control at phase
            u_k = ca.MX.sym(self.ocp.u.name() + '_' + str(phase_k))
            # append controls
            self.nlp['z'].append(u_k)
            self.nlp['lbz'].append(self.ocp.u.lb())
            self.nlp['ubz'].append(self.ocp.u.ub())
            self.nlp['z0'].append(self.ocp.u.lb())
            self.plot['u'].append(u_k)

            # add symbolic variables for phase k collocation points
            y_c = []
            for tau_j in range(degree):
                # boundary initial state condition (initial boundary condition)
                y_k_j = ca.MX.sym(self.ocp.y.name() + '_' + str(phase_k) + '_' + str(tau_j), self.ocp.y.dim())
                y_c.append(y_k_j)
                self.nlp['z'].append(y_k_j)
                self.nlp['lbz'].append(self.ocp.y.lb())
                self.nlp['ubz'].append(self.ocp.y.ub())
                self.nlp['z0'].append(self.ocp.y.lb())

            # assign equality constraints for polynomial interpolation
            y_k_end = self.D[0] * y_k
            for tau_j in range(degree):
                y_p = self.C[0, tau_j + 1] * y_k
                for tau_r in range(degree): y_p += self.C[tau_r + 1, tau_j + 1] * y_c[tau_r]

                if self.ocp.has_DAE():
                    # TODO handle DAE
                    f_tau_j, g_tau_j, w_tau_j = self.ocp.ocp_fn(y_c[tau_j], u_k)
                else:
                    f_tau_j, w_tau_j = self.ocp.ocp_fn(y_c[tau_j], u_k)

                self.nlp['g'].append(h * f_tau_j - y_p)
                self.nlp['lbg'].append(np.zeros(self.ocp.y.dim()))
                self.nlp['ubg'].append(np.zeros(self.ocp.y.dim()))

                y_k_end += self.D[tau_j + 1] * y_c[tau_j]

                self.nlp['J'] += self.B[tau_j + 1] * w_tau_j * h

            y_k = ca.MX.sym(self.ocp.y.name() + '_' + str(phase_k + 1), self.ocp.y.dim())
            self.nlp['z'].append(y_k)
            self.nlp['lbz'].append(self.ocp.y.lb())
            self.nlp['ubz'].append(self.ocp.y.ub())
            self.nlp['z0'].append(self.ocp.y.lb())
            self.plot['y'].append(y_k)

            # enforce continuity constraint
            self.nlp['g'].append(y_k_end - y_k)
            self.nlp['lbg'].append(np.zeros(self.ocp.y.dim()))
            self.nlp['ubg'].append(np.zeros(self.ocp.y.dim()))

        # Concatenate vectors
        self.nlp['z'] = ca.vertcat(*self.nlp['z'])
        self.nlp['g'] = ca.vertcat(*self.nlp['g'])
        self.plot['y'] = ca.horzcat(*self.plot['y'])
        self.plot['u'] = ca.horzcat(*self.plot['u'])
        self.nlp['z0'] = np.concatenate(self.nlp['z0'])
        self.nlp['lbz'] = np.concatenate(self.nlp['lbz'])
        self.nlp['ubz'] = np.concatenate(self.nlp['ubz'])
        self.nlp['lbg'] = np.concatenate(self.nlp['lbg'])
        self.nlp['ubg'] = np.concatenate(self.nlp['ubg'])

        # construct nlp solver
        prob = {'f': self.nlp['J'], 'x': self.nlp['z'], 'g': self.nlp['g']}
        self.nlp['solver'] = ca.nlpsol('solver', 'ipopt', prob)

    def solve_nlp(self, *y_0):
        # enforce initial conditions
        self.nlp['z0'][0:self.ocp.y.dim()] = y_0
        self.nlp['lbz'][0:self.ocp.y.dim()] = y_0
        self.nlp['ubz'][0:self.ocp.y.dim()] = y_0

        # call defined solver
        solution = self.nlp['solver'](x0=self.nlp['z0'], lbx=self.nlp['lbz'], ubx=self.nlp['ubz'], lbg=self.nlp['lbg'],ubg=self.nlp['ubg'])

        # overwrite z0 for hot start in next nlp solution
        if self.nlp_config['warm_start']:
            self.nlp['z0'] = np.array(solution['x']).flatten()

        # parse trajectories
        y_opt, u_opt = ca.Function('trajectories', [self.nlp['z']], [self.plot['y'], self.plot['u']], ['z'], ['y', 'u'])(solution['x'])

        return y_opt.full(), u_opt.full()


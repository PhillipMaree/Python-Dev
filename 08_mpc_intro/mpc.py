import matplotlib.pyplot as plt
from nlp import NLP
import numpy as np

''' NMPC implementation '''


class NMPC(NLP):

    def __init__(self, td, tf, N):
        NLP.__init__(self, td, N)

    def simulate(self, *x_0):
        x_opt, u_opt = self.solve(*x_0)
        return x_opt, u_opt


if __name__ == '__main__':

    # simulation interval
    tf = 10
    # prediction horizon
    N = 30
    # sampling rate
    td = 0.1
    # initial state

    # solve NMPC for closed-loop operation
    x_0 = [0.0,0.0,0.0]
    x_opt, u_opt = NMPC(td, tf, N).simulate(*x_0)

    # Plot the result
    tgrid = np.linspace(0, tf, N+1)
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x_opt[0], '--')
    plt.plot(tgrid, x_opt[1], '-')
    plt.plot(tgrid, x_opt[2], '-')
    plt.xlabel('t')
    plt.legend(['u1'])
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.clf()
    plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
    plt.xlabel('t')
    plt.legend(['x1','x2','x3'])
    plt.grid()
    plt.show()
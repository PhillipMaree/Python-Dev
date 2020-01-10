import matplotlib.pyplot as plt
from nlp import NLP
import numpy as np

''' NMPC implementation '''


class NMPC(NLP):

    def __init__(self, td, tf, N):
        NLP.__init__(self, td, N)

    def simulate(self, *y_0):
        return self.solve(*y_0)


if __name__ == '__main__':

    # simulation interval
    tf = 10
    # prediction horizon
    N = 30
    # sampling rate
    td = 0.1
    # initial state

    # solve NMPC for closed-loop operation
    y_0 = [0.1,0.1,0.1]
    y_opt, u_opt = NMPC(td, tf, N).simulate(*y_0)

    # Plot the result
    tgrid = np.linspace(0, tf, N+1)
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, y_opt[0], '--')
    plt.plot(tgrid, y_opt[1], '-')
    plt.plot(tgrid, y_opt[2], '-')
    plt.xlabel('t')
    plt.legend(['y1','y2','y3'])

    plt.figure(2)
    plt.clf()
    plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
    plt.xlabel('t')
    plt.legend(['u1'])
    plt.grid()
    plt.show()
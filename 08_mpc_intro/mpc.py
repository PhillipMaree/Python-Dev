import matplotlib.pyplot as plt
from nlp import NLP
import numpy as np
import pandas as pd

''' NMPC implementation '''


class NMPC(NLP):

    def __init__(self, td, tf, N):
        self.tf = tf
        NLP.__init__(self, td, N)

    def run(self, *y_0):
        df = pd.DataFrame(columns=['time', 'y1', 'y2', 'y3', 'u'])
        for tk in range(round(tf/td)+1):
            y_opt, u_opt = self.solve_nlp(*y_0)
            df.loc[tk] = np.append(tk*td, np.append(y_opt[:,0], u_opt[:,0]))
            y_0 = y_opt[:,1]
        return df


if __name__ == '__main__':

    # simulation interval
    tf = 3
    # prediction horizon
    N = 10
    # sampling rate
    td = 0.1
    # initial state

    # solve NMPC for closed-loop operation
    y_0 = [0.25, 0.25, 0.25]
    df = NMPC(td, tf, N,).run(*y_0)

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    df.plot(x='time', y='y1', ax=ax1)
    ax2 = fig.add_subplot(222)
    df.plot(x='time', y='y2', ax=ax2)
    ax3 = fig.add_subplot(223)
    df.plot(x='time', y='y3', ax=ax3)
    ax4 = fig.add_subplot(224)
    df.plot(x='time', y='u', ax=ax4,drawstyle="steps")
    plt.show()

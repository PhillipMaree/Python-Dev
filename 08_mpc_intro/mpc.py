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
    y_0 = [0.1, 0.2, 0.3]
    df = NMPC(td, tf, N,).run(*y_0)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    df.plot(x='time', y=['y1','y2','y3'], ax=ax1)

    ax2 = fig.add_subplot(212)
    df.plot(x='time', y='u', ax=ax2,drawstyle="steps")
    plt.show()

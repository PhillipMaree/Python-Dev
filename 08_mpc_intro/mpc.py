import matplotlib.pyplot as plt
from nlp import NLP
import numpy as np
import pandas as pd

''' NMPC implementation '''


class NMPC(NLP):

    def __init__(self, h, N, tf):
        self.tf = tf
        NLP.__init__(self, h, N)

    def run(self, *y_0):

        df_cl = pd.DataFrame(columns=['time', 'y1', 'y2', 'y3', 'u'])
        for tk in range(round(tf/h)+1):
            df_ol = self.solve_ocp(tk, *y_0)
            return df_ol
            y_0 = np.array(df_ol.loc[1,['y1','y2','y3']])
            if tk==0:
                df_cl.loc[0] = df_ol.loc[0,['time', 'y1', 'y2', 'y3', 'u']]
            df_cl.loc[tk] = df_ol.loc[tk,['time', 'y1', 'y2', 'y3', 'u']]

        return df_cl


if __name__ == '__main__':

    # simulation interval
    tf = 0.1
    # prediction horizon
    N = 10
    # sampling rate
    h = 0.1
    # initial state

    # solve NMPC for closed-loop operation

    y_0 = [0.2, 0.3, 0.4]
    df = NMPC(h, N, tf).run(*y_0)

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


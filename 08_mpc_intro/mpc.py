import matplotlib.pyplot as plt
from nlp import NLP
import numpy as np
import pandas as pd

import casadi as ca

''' NMPC implementation '''


class NMPC(NLP):

    def __init__(self, h, N, tf):
        self.tf = tf
        self.nlp_config = {'h': h, 'N': N, 'collocation_method': 'radau', 'collocation_degree': 6}
        NLP.__init__(self, self.nlp_config)

    def run(self, *y_0):

        df_cl = pd.DataFrame(columns=['time', 'y1', 'y2', 'y3', 'u'])
        for tk in range(round(tf/h)+1):
            # solve for open-loop
            df_ol = self.solve_ocp(tk, *y_0)
            # build closed-loop evolutions
            df_cl = df_cl.append( df_ol.loc[0:self.nlp_config['collocation_degree'] ,['time', 'y1', 'y2', 'y3', 'u']], ignore_index=True )

            # "simulate" for next initial condition
            y_0 = np.array(df_ol.loc[1+self.nlp_config['collocation_degree'],['y1','y2','y3']])
        return df_cl

if __name__ == '__main__':

    tf = 2.5                         # simulation interval
    N = 10                          # prediction horizon
    h = 0.25                         # sampling rate
    y_0 = [0.1, 0.1, 0.1]           # initial state

    # solve NMPC for closed-loop operation
    df = NMPC(h, N, tf).run(*y_0)

    # plotting closed-loop results
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


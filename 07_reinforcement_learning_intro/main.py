import numpy as np
import pandas as pd
import random


class RL:

    def __init__(self,environment):
        self.alpha = 0.8
        self.sigma = 0.1
        self.explore = 0.9
        self.qt = pd.DataFrame(100,index=environment.get_actions(), columns=environment.get_states())

    def max_reward_action(self,state):
        return self.qt[state].idxmax()

    def Q(self,state):
        return self.qt.loc[:,state].max()

    def q_learning(self,state):

        action = self.qt[state].idxmax()
        if np.random.random() < self.explore:
            action = self.qt.index[random.randint(0,3)]

        terminate, reward, next_state = environment.action(action,state)

        #print("State %d Action %s Next State %d Reward %d T %d" % (state, action, next_state, reward, terminate))

        if terminate:
            self.qt.loc[action,state] = reward
        else:
            self.qt.loc[action,state] += self.alpha*(reward + self.sigma*self.Q(next_state) - self.qt.loc[action,state])

        return terminate, next_state, self.qt

class Agent:

    def __init__(self):
        pass


class Environment:

    MAX_PENALTY = -1000
    MAX_REWARD = 8

    def __init__(self):
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.states = [11, 12, 13, 14, 15,21, 22, 23, 24, 25,31, 32, 33, 34, 35,41, 42, 43, 44, 45,51, 52, 53, 54, 55]
        self.rewards = pd.DataFrame([[0, 1, 2, 3, 4],[1, 2, 3, 4, 5],[2, 3, 4, 5, 6],[3, 4, 5, 6, 7],[4, 5, 6, 7, 8]], index={1,2,3,4,5}, columns={1,2,3,4,5} )

    def get_actions(self):
        return self.actions

    def get_states(self):
        return self.states

    def action(self,action, state):
        row,col = round(state/10),round(state%10)

        old_reward = self.rewards.loc[row,col]
        if action=='Up':
            row -= 1
        elif action=='Down':
            row += 1
        elif action=='Left':
            col -= 1
        else:
            col += 1
        new_state = row*10+col

        terminate = False
        if col < 1 or col >5 or row < 1 or row > 5:
            reward = -1000
            terminate = True
        elif col == 5 and row == 5:
            reward = self.MAX_REWARD
            terminate = True
        else:
            reward =  self.rewards.loc[row,col]
            #if reward<old_reward:
            #    reward = -1000

        return terminate, reward, new_state

if __name__ == '__main__':
    print("Reinforcement Learning Introduction")

    environment = Environment()
    rl = RL(environment)

    terminate = True
    episodes = 5000
    state_init = 11


    while episodes>0:

        if terminate:
            state = state_init

        terminate, state, q = rl.q_learning(state)



        episodes -= 1

    print(q)
    print(state)

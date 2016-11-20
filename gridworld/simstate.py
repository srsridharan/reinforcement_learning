"""
This file simulates the state transitions in a gridworld
"""
import numpy as np


class dynamics:
    def __init__(self, ndim=2, gridmax=10, target_state=None):
        self.x = np.zeros((ndim, 1))
        self.__A = np.eye(ndim)
        self.__B = np.eye(ndim)
        self.lim = gridmax-1
        self.__targetstate = target_state
        self.xstate = tuple(self.x.flatten().astype('int'))
        self.ndim = ndim

    def nextState(self, action_now, state_now):
        self.x = np.reshape(state_now, (self.ndim, 1))
        self.xstate = state_now
        if self.xstate == self.__targetstate:
            # absorbing state
            pass
        else:
            xnext = self.__A.dot(self.x) + self.__B.dot(np.reshape(action_now,
                                                                   (self.ndim,
                                                                    1)))

            if((0<=xnext[0]<=self.lim) & (0<=xnext[1]<=self.lim)):
                self.x = xnext

            self.xstate = tuple(self.x.flatten().astype('int'))
        return self.xstate


def stateActionToRewardMap(state, action):
    rval = rewardmap.get((state, action))
    return rval


def createRewardMap(targetstate, actionset):
    return {(targetstate, a): 1 for a in actionset}


def createStateSpace(gridMax):
    x1 = range(gridMax)
    X, Y = np.meshgrid(x1, x1)
    return zip(X.flatten(), Y.flatten())

if __name__ == "__main__":
    targetstate = (5, 5)
    actionset = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0))
    gridMax = 10

    dyna = dynamics(gridmax=gridMax, targetstate=targetstate)
    rewardmap = createRewardMap(targetstate, actionset)
    statespace = createStateSpace(gridMax)

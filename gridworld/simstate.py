"""
This file simulates the state transitions in a gridworld
"""
import numpy as np


'''
class stateTransDeterministic:
    def __init__(self, x_zero=None, A=None, B=None,  numGridPts=10, dimstatespace=2):
        """
        @x_0: initial state value
        @numGridPts: the grid world extends to +- numGridPts

        """
        if x_zero == None:
            self.x_zero = tuple([0]*dimstatespace)
            self.x = self.x_zero

        if A == None:
            A = np.eye((dimstatespace, dimstatespace))
            B = np.eye((dimstatespace, dimstatespace))
        self.numGridPts = numGridPts
        self.A = A
        self.B = B


    def stateNext(self, action):
        x_kplus1 = self.A*self.x+  B*u

'''


'''
class stateTransition:
    def __init__(self, x_zero=None, A=None, B=None,  numGridPts=10, dimstatespace=2):
        """
        @x_0: initial state value
        @numGridPts: the grid world extends to +- numGridPts

        """
        if x_zero == None:
            self.x_zero = tuple([0]*dimstatespace)
            self.x = self.x_zero

        if A == None:
            A = np.eye((dimstatespace, dimstatespace))
            B = np.eye((dimstatespace, dimstatespace))
        self.numGridPts = numGridPts
        self.A = A
        self.B = B
'''

targetstate=(5,5)
actionset = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0))
gridMax = 10

def createRewardMap(targetstate, actionset):
    return {(targetstate,a):1 for a in actionset}

def create_initial_policy(stateSpace):
    policy = dict()
    for k in stateSpace:
        policy[k] = actionset[np.random.randint(0, len(actionset))]
    return policy


class dynamics:
    def __init__(self, ndim=2, gridmax=10):
        self.x = np.zeros((ndim, 1))
        self.__A = np.eye(ndim)
        self.__B = np.eye(ndim)
        self.lim = gridmax
        self.xstate = tuple(self.x.flatten().astype('int'))

    def nextState(self, control_input):
        if self.x == targetstate:
            #absorbing state
            pass
        else:
            xnext = self.__A.dot(self.x) + self.__B.dot(np.reshape(control_input, (self.ndim, 1)))

            if((-self.lim<=xnext[0]<=self.lim) & (-self.lim<=xnext[1]<=self.lim)):
                self.x = xnext

            self.xstate = tuple(self.x.flatten().astype('int'))


def stateActionToRewardMap(state, action):
    rval = rewardmap.get((state, action))
    return rval


def createStateSpace(gridMax):
    x1 = range(gridMax)
    X, Y = np.meshgrid(x1, x1)
    return zip(X.flatten(), Y.flatten())

dyna = dynamics(gridmax=gridMax)
rewardmap = createRewardMap(targetstate, actionset)
statespace = createStateSpace(gridMax)
policy = create_initial_policy(statespace)

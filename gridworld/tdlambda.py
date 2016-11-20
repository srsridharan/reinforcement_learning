"""
This file implements the TD(0) i.e. SARSA approach
"""
import numpy as np
import simstate as simstate
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # needed for the 3d projection
import matplotlib.pyplot as plt


target_state = (5, 5)
actionset = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0))
gam_var = 0.7
alpha_var = 0.05
gridMax = 10
default_q = 2
lambda_var = .4


dyna = simstate.dynamics(gridmax=gridMax, target_state=target_state)
statespace = simstate.createStateSpace(gridMax)
Q_val = {(target_state, a): 1 for a in actionset}


class TDimplementation:
    def __init__(self, alpha=0.0, discount_gam_var=0.0, Q_val=None, default_q=2,
                 lambda_param=0.4, epsilon=.2, fn_reward_valuator=None):
        self.gamma = discount_gam_var
        self.alpha = alpha
        self.Qval = Q_val
        self.reward_fn = fn_reward_valuator
        self.default_q = default_q
        self.epsilon = epsilon
        self.__eligibilitybuffer = dict()
        self.__stateactionstatequartetbuffer = list()
        self.lambda_param = lambda_param

    def Q_iter(self, state_now, action_now, state_next, action_next):
        """
        The action value function update via TD(lambda) is done as follows
        Repeat for each step in the episode
        * \delta :=  r(s, a , s') + \gamma Q(s', a') - Q(s, a)
        * \eligibility(s, a) <- \eligibility(s, a) + 1
        * for all (s, a) on this trail (i.e. in state-action pair buf):
                Q(s,a) <- Q(s, a) +  \alpha \delta eligibility(s, a)
                \eligibility(s, a) <- \lambda \eligibility(s, a)

        """
        self.__stateactionstatequartetbuffer.append(
            {'state_now': state_now, 'action_now': action_now,
             'state_next': state_next, 'action_next': action_next})

        # \delta :=  r(s, a , s') + \gamma Q(s', a') - Q(s, a)
        difference = self.reward_fn(state_now, action_now, state_next) +\
            self.gamma*self.Qval.get((state_next, action_next),
                                     np.abs(np.random.normal(0, 0.001))) -\
            self.Qval.get((state_now, action_now),
                          np.abs(np.random.normal(0, 0.001)))

        # \eligibility(s, a) <- \eligibility(s, a) + 1
        self.__eligibilitybuffer[(state_now, action_now)] = \
            self.__eligibilitybuffer.get((state_now, action_now), 0) + 1

        # for each step of the episode do the following
        for quartet in self.__stateactionstatequartetbuffer:
            state_action_now = (quartet['state_now'], quartet['action_now'])
            random_positive_float = np.abs(np.random.normal(0, 0.001))

            # Q_update
            self.Qval[state_action_now] = self.Qval.get(state_action_now, random_positive_float) +\
                self.alpha *\
                difference*self.__eligibilitybuffer.get(state_action_now, 0)

            # Eligibility decay
            self.__eligibilitybuffer[state_action_now] = \
                self.lambda_param*self.__eligibilitybuffer.get(state_action_now,
                                                               0)

    def generate_action(self, state_now):
        Qval = self.Qval
        action_vals = {Qval.get((state_now, act), self.default_q): act for act
                       in actionset}
        best_action = action_vals.get(max(action_vals))
        if np.random.uniform(0, 1) < self.epsilon:
            randomaction = actionset[np.random.randint(0, len(actionset))]
            return randomaction
        else:
            return best_action

    def update_epsilon(self):
        self.epsilon = self.epsilon/(1+self.epsilon)

    def reset_episode(self):
        # this resets the eligibility trace. it is invoked after each episode
        # i.e once the target state is reached
        self.__eligibilitybuffer = dict()
        self.__stateactionstatequartetbuffer = list()


def reward_function(state_now, action_now, state_next):
    """
    inputs:
        current_state
        current_action
        state_next
    output:
        reward (scalar)
    """
    if state_next == target_state:
        return 3
    else:
        return -0.2


def plot_results(gridMax, Q1, A1):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(range(0, gridMax), range(0, gridMax))
    surf = ax.plot_surface(X, Y, Q1, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def run_iteration(tdObject):
    """
    input: tdlambda object
    returns: dict with keys ('Qval_optimal',  'action_optimal')
    """
    for k in xrange(2000):
            # select random starting state
            state_now = statespace[np.random.randint(0, gridMax**2)]
            action_now = tditer.generate_action(state_now)
            # while state is not target_state do
            while state_now != target_state:
                state_next = dyna.nextState(action_now, state_now)
                action_next = tditer.generate_action(state_next)
                tditer.Q_iter(state_now, action_now, state_next, action_next)
                state_now = state_next
                action_now = action_next
            tditer.reset_episode()
            tditer.update_epsilon()

    # now compute the max value function
    Q1 = np.zeros((gridMax, gridMax))
    A1 = np.zeros((gridMax, gridMax))

    for s in statespace:  # TODO: can be refactored more elegantly using argmax
        value_dict_for_state = {tditer.Qval.get((s, actionset[k]), -.1): k for k
                                in range(len(actionset))}
        keymax = max(value_dict_for_state)
        Q1[s[0], s[1]] = keymax
        A1[s[0], s[1]] = value_dict_for_state[keymax]

    return {'Qval_optimal': Q1, 'action_optimal': A1}

if __name__ == "__main__":
    tditer = TDimplementation(Q_val=Q_val, default_q=default_q,
                              lambda_param=lambda_var, alpha=alpha_var,
                              discount_gam_var=gam_var,
                              fn_reward_valuator=reward_function)
    optval_dict = run_iteration(tditer)

    plot_results(gridMax,
                 optval_dict['Qval_optimal'], optval_dict['action_optimal'])

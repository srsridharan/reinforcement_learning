"""
In this function we demonstrate off policy learning (via Q learning).
This involves learning an optimal strategy when the policy being evaluated is
different from the policy being employed to select the actions.
"""

import numpy as np
import simstate as simstate
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # needed for the 3d projection
import matplotlib.pyplot as plt

target_state = (5, 5)
actionset = ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0))
gam_var = 0.4
alpha_var = 0.03
gridMax = 10
default_q = 2

dyna = simstate.dynamics(gridmax=gridMax, target_state=target_state)
# rewardmap = simstate.createRewardMap(target_state, actionset)
statespace = simstate.createStateSpace(gridMax)


class QTDimplementation:
    def __init__(self, alpha=0.0, discount_gam_var=0.0, Q_val=None, default_q=2,
                 epsilon=1, fn_reward_valuator=None):
        self.gamma = discount_gam_var
        self.alpha = alpha
        self.Qval = Q_val
        self.reward_fn = fn_reward_valuator
        self.default_q = default_q
        self.epsilon = epsilon

    def Q_iter(self, state_now, action_now, state_next, action_next):
        # The policy iteration takes the form
        # Q(s,a)<- Q(s, a) +  \alpha [R(s,a,s') + \gamma max_{a'} Q(s', a') - Q(s, a)]
        Qmaxaprime = max([self.Qval.get((state_next, action), np.abs(np.random.normal(0,.001))) for action in actionset ]) #TODO can be made more efficient by only
        # lookign at actions available at that state and by enabling and storing
        # faster max lookup
        difference = self.reward_fn(state_now, action_now, state_next) + self.gamma*Qmaxaprime - \
            self.Qval.get((state_now, action_now), np.abs(np.random.normal(0,.001)))
        self.Qval[(state_now, action_now)] = self.Qval.get(
            (state_now, action_now), np.abs(np.random.normal(0,.001))) + self.alpha * difference

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
        # self.epsilon = self.epsilon


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
        return 0
    else:
        return 0


def plot_results(gridMax, Q1, A1):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(range(0, gridMax), range(0, gridMax))
    surf = ax.plot_surface(X, Y, Q1, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show(block=False)


def run_qtditer(qtditer_object, nsteps=800):
    '''
    run the td iteration  for @nsteps  steps on a TD iterator object
    @qtditer_object

    input:
        @qtditer_object
        @nsteps: number of restarts
    '''

    for k in xrange(nsteps):
        # select random starting state
        state_now = statespace[np.random.randint(0, 100)]
        action_now = qtditer.generate_action(state_now)
        # while state is not target_state do
        while state_now != target_state:
            state_next = dyna.nextState(action_now, state_now)
            action_next = qtditer_object.generate_action(state_next)
            qtditer_object.Q_iter(state_now, action_now, state_next, action_next)

            state_now = state_next
            action_now = action_next

    # now compute the max value function
    Q1 = np.zeros((gridMax, gridMax))
    A1 = np.zeros((gridMax, gridMax))

    for s in statespace:  # TODO: can be refactored more elegantly using argmax
        value_dict_for_state = {qtditer_object.Qval.get((s, actionset[k]), -.1): k for k
                                in range(len(actionset))}
        keymax = max(value_dict_for_state)
        Q1[s[0], s[1]] = keymax
        A1[s[0], s[1]] = value_dict_for_state[keymax]

    return {'Qval_optimal': Q1, 'action_optimal': A1}


if __name__ == "__main__":
    Q_val = {(target_state, a): 1 for a in actionset}
    qtditer = QTDimplementation(Q_val=Q_val, default_q=default_q, alpha=alpha_var,
                                discount_gam_var=gam_var,
                                fn_reward_valuator=reward_function)
    qtd_iter_results = run_qtditer(qtditer, nsteps=1400)
    plot_results(gridMax, qtd_iter_results['Qval_optimal'],
                 qtd_iter_results['action_optimal'])


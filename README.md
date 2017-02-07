Reinforcement learning 
========
This is a collection of folders on the topic of reinforcemnt learning



## Contents

- gridworld (the grid world example, you can move up/down/left/right to reach a target)
	- sarsa.py (using TD(0) control)
	- simstate.py (the base file to model the dynamics of the one step markov chain as a state space system with 2 inputs and 2 states)
	- tdlambda.py (the TD(\lambda) approach to control)

## Figures
These are sample figures to indicate the optimal cost functions from the various algorithms on the gridworld example

- SARSA 
![alt text][sarsafig]
[sarsafig]: https://github.com/srsridharan/reinforcement_learning/blob/master/gridworld/plots/sarsa.png "SARSA optimal cost value function for gridworld"
-  TD(lambda)
![alt text][tdlambdafig]
[tdlambdafig]: https://github.com/srsridharan/reinforcement_learning/blob/master/gridworld/plots/tdlambda.png "TD(lambda) optimal cost value function for gridworld"


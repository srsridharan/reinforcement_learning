# Outline

This directory contains algorithm implementations for the gridworld problem. A system evolves deterministically  and the goal is to reach the destination while being able  to control the evolution of the system by pushing the state to move up/down/left/right by one step each time. There are several algorithms described herein to obtain the optimal policy (ref Contents section below). For a more indepth insight into the algoritms refer to [the introduction to reinforcement learning document](https://github.com/srsridharan/smlNotes/blob/master/researchcode/RLnotes/rl_notes.pdf)

## Contents


- simstate.py: the simulator for the evolution of the state space modeling the markov chain whose evolution is being controlled.
- sarsa.py: The SARSA on-policy RL algorithm (TD(0) control).
- tdlambda.py: this implements the TD(lambda) algorithm for control using the  eligibility curve idea. 
- plots: contains some of the plots of the results for the optimal cost value function/ actions obtained.
	- sarsa.png: for SARSA


## How to run 

- SARSA.py
```python
cmd> ipython
>> %run sarsa.py
```

![alt text][sarsafig]
[sarsafig]: https://github.com/srsridharan/reinforcement_learning/blob/master/gridworld/plots/sarsa.png "SARSA optimal cost value function for gridworld"


- tdlambda.py
```python
cmd> ipython
>> %run tdlambda.py
```
![alt text][tdlambdafig]
[tdlambdafig]: https://github.com/srsridharan/reinforcement_learning/blob/master/gridworld/plots/tdlambda.png "TD(lambda) optimal cost value function for gridworld"


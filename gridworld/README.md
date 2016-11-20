# Outline

This dicrectory contains algorithm implementations for the gridworld problem. A system evolves determinisically  and the goal is to reach the destination. There are several algorithms described herein (ref contents).

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
[sarsafig]: https://github.com/srsridharan/reinforcement_learning/blob/master/gridworld/plots/sarsa.png "SARSA optimal cost value function"


- tdlambda.py
```python
cmd> ipython
>> %run tdlambda.py
```

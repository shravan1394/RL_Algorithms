import numpy as np
from BanditArmClass import *
from ValFuncOptimizer import *
from PacOptimizer import *

N = 100000;
states = 00;
problems = 2000
eps = 0.0;
tau = 0.2;
epsi = 2.0;
delta = 1.0
#soft_cache = (tau); # use this cache for softmax mode 
#eps_cache = (eps);# use this cache for eps-greedy mode

dist_cache = (1, 0, 0, 1); # (meanDist_mean meanDist_Var, varDist_Mean, varDist_Var) for the bandit arms 

avg_reward = np.zeros(states);
optim_action = np.zeros(states);
Q = np.zeros((problems, N));
Na = np.zeros((problems, N));

X = np.random.randn; # sample

Arms = BanditArm(dist_cache, N);

action = Arms.initiateTestBed(problems);
action_cache = (Arms); 
update(Q, Na, action, action_cache, rule = 'mean');

for i in range(1,states):
	if i % 100 == 0:
		print(i)
	UCB1_cache = (i, Na); # use this cache for UCB1 mode

	action = nextAction(Q, UCB1_cache, policy = 'UCB1');

	update(Q, Na, action, action_cache, rule = 'mean');

	# Find avg reward and % of optim action take for each problem in all states
	avg_reward[i] = Q[np.arange(problems), action].mean();
	optim_action[i] = (action == Arms.AStar).mean();

## PAC Optimal arm
opti_arm = PacArm(epsi, delta, action_cache, mode = 'naive');
print(opti_arm == Arms.AStar);
opti_arm = PacArm(epsi, delta, action_cache, mode = 'median');
print(opti_arm == Arms.AStar);
Arms.plot(avg_reward, optim_action);


import gym
import numpy as np
from DP_Solns_RL import *
from MDPModel import *

#env = gym.make('FrozenLake-v0');
env = gym.make('FrozenLake8x8-v0');

n = env.env.ncol;
gamma = 0.9;
eps = 0.0001;

success = 0;
episodes  = 1000;
timeSteps = 1000;
goal = n * n - 1;

# Rewards for 'FrozenLake'
G = 100;
S = F = 0;
H = -100;

imR = np.array([[S,F,F,F,F,F,F,F,
                 F,F,F,F,F,F,F,F,
                 F,F,F,H,F,F,F,F,
                 F,F,F,F,F,H,F,F,
                 F,F,F,H,F,F,F,F,
                 F,H,H,F,F,F,H,F,
                 F,H,F,F,H,F,H,F,
                 F,F,F,H,F,F,F,G]]);



"""

imR = np.array([[S,F,F,F,
                 F,H,F,H,
                 F,F,F,H,
                 H,F,F,G]]);


"""
MDP = MDPModel(env.env, imR, gamma, eps);

DPSolve = DP_Solns_RL(MDP);

#DPSolve.PolicyIteration();
#DPSolve.ValueIteration();
for i_episode in range(episodes):
	observation = env.reset();

	for t in range(timeSteps):

		act = DPSolve.action(observation);
		observation, reward, done, info = env.step(act);
		if done:
			if observation == goal:
				success += 1;
				print("congrats!! you achieved after " + str(i_episode) + " episodes");
			break;
		DPSolve.ValueIteration("RTDP", (0.0, observation));

	if(t == timeSteps - 1):
		print("Reached {} steps without falling".format(timeSteps));
		break;

env.close();

print("You will reach your goal with a {}% success".format((success / episodes) * 100));
print(DPSolve.Policy.reshape(n, n));

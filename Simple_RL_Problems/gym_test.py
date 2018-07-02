import gym
import numpy as np
from DP_Solns_RL import *

#env = gym.make('FrozenLake-v0');
env = gym.make('FrozenLake8x8-v0');

n = 8; # change n = 4 for 'FrozenLake-v0'
gamma = 0.999;
eps = 0.0001;


success = 0;
episodes  = 50000;
timeSteps = 500;
goal = n * n - 1;


# Rewards for 'FrozenLake8x8-v0'
G = 20000;
S = F = 0;
H = -15000;

imR = np.array([[S,F,F,F,F,F,F,F,
                 F,F,F,F,F,F,F,F,
                 F,F,F,H,F,F,F,F,
                 F,F,F,F,F,H,F,F,
                 F,F,F,H,F,F,F,F,
                 F,H,H,F,F,F,H,F,
                 F,H,F,F,H,F,H,F,
                 F,F,F,H,F,F,F,G]]);



"""
# Rewards for 'FrozenLake-v0'
G = 170;
S = F = 0;
H = -100;

imR = np.array([[S,F,F,F,
                 F,H,F,H,
                 F,F,F,H,
                 H,F,F,G]]);
"""


def TrProbGen(direction):
	
	p = np.zeros((n * n, n * n));	
	if direction == 0:	
		Pleft = 1/3;Pdown = 2/9;Pright = 2/9;Pup = 2/9;
	elif direction == 1:	
		Pdown = 1/3;Pleft = 2/9;Pright = 2/9;Pup = 2/9;
	elif direction == 2:	
		Pright = 1/3;Pdown = 2/9;Pleft = 2/9;Pup = 2/9;
	elif direction == 3:	
		Pup = 1/3;Pdown = 2/9;Pright = 2/9;Pleft = 2/9;
	for i in range(n):
		for j in range(n):
			if i - 1 >= 0:
				p[n * i + j, n * (i - 1) + j] += Pup;
			else:
				p[n * i + j, n * i + j] += Pup;

			if j - 1 >= 0:
				p[n * i + j, n * i + j - 1] += Pleft;
			else:
				p[n * i + j, n * i + j] += Pleft;

			if i + 1 < n:
				p[n * i + j, n * (i + 1) + j] += Pdown;
			else:
				p[n * i + j, n * i + j] += Pdown;

			if j + 1 < n:
				p[n * i + j, n * i + j + 1] += Pright;
			else:
				p[n * i + j, n * i + j] += Pright;

	return p;




trProb = {0:TrProbGen(0),1:TrProbGen(1),2:TrProbGen(2),3:TrProbGen(3)};

expR = {0:trProb[0].dot(imR.T),1:trProb[1].dot(imR.T),2:trProb[2].dot(imR.T),3:trProb[3].dot(imR.T)};

DPSolve = DP_Solns_RL(n, trProb, expR, eps, gamma);

#DPSolve.ValueIteration("Vectorized");

for j in range(1):
	for i_episode in range(episodes):
		observation = env.reset();
		for t in range(timeSteps):
			
			act = DPSolve.action(observation);
			observation, reward, done, info = env.step(act);
			DPSolve.ValueIteration("RTDP", observation);
			if done:
				
				if observation == goal:
					success += 1;
					print("congrats!! you achieved after " + str(i_episode) + " episodes");
					#env.render();
				#else:
					#print("uh oh!!. You fell :(");
				break;
		if(t == timeSteps - 1):
			print("Reached {} steps without falling".format(timeSteps));

env.close();

print("You will reach your goal with a {}% success".format((success / episodes) * 100));  

#print(DPSolve.Policy);
#print(DPSolve.V);

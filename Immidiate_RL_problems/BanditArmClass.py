# Test Bed
import numpy as np
import matplotlib.pyplot as plt

class BanditArm:
	
	def __init__(self, cache = (1, 0, 0, 1), N = 10):
		
		self.meanDistM, self.meanDistV, self.varDistM, self.varDistV = cache;

		self.mean = self.meanDistM * np.random.randn(N) + self.meanDistV * np.ones(N);
		self.var = self.varDistM * np.random.randn(N) + self.varDistV * np.ones(N);
		self.N = N;
		self.X = np.random.randn;
		self.AStar = np.argmax(self.mean);

	def initiateTestBed(self, problems):
		
		self.problems = problems;
		action = np.random.randint(self.N, size = problems);
		
		return action;

	def pullArm(self, N = None, problems = None, action = None):

		if N == None:
			N = self.N;

		if problems == None:
			R = self.var[action] * self.X(self.problems) + self.mean[action];

		else: 
			R = self.var[action] * self.X(problems, N) + self.mean[action]; 

		return R;

	def plot(self, reward, action):
		
		f = plt.figure(1);
		plt.plot(reward);
		g = plt.figure(2);
		plt.plot(action);
		g.show();
		f.show();
		raw_input();



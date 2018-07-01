"""
Dynamic Programming solutions for simple RL problems
"""
import numpy as np


class DP_Solns_RL:

	def __init__(self, n, trProb, expR, eps, gamma):
		
		self.n = n;
		self.trProb = trProb;
		self.expR = expR;
		self.eps = eps;
		self.gamma = gamma;
		self.V = np.random.rand(n * n, 1); 
		self.Policy = np.array([]);

	def ValueIteration(self, mode = "Iterative"):
	
	

		if mode == "Iterative":
			while True:
	
				L = np.array([self.expR[i] + self.gamma * self.trProb[i].dot(self.V) for i in range(4) ]).reshape(4,self.n * self.n);
				Vn = np.max(L,0).T[:,None];
				self.Policy = np.argmax(L,0).T[:,None];
				print (np.linalg.norm(Vn-self.V));
				if np.linalg.norm(Vn-self.V) < self.eps * (1 - self.gamma)/(2 * self.gamma):
					self.V = Vn;
					break;
				else:
					self.V = Vn;


	def action(self, observation):
		
		return int(self.Policy[observation]);

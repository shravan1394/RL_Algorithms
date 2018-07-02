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
		self.Policy = np.random.randint(4, size = (n * n, 1));

	def ValueIteration(self, mode = "Iterative", cache = None):
	
	

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

		if mode == "RTDP":
			
			obs = cache;
			L = np.array([self.expR[i][obs] + self.gamma * self.trProb[i][obs][None, :].dot(self.V) for i in range(4) ]).reshape(4, 1);

			self.V[obs] = np.max(L,0);
			self.Policy[obs] = np.argmax(L,0);

		if mode == "Vectorized": # still working on this!!

			L = np.array([np.linalg.inv(np.identity(self.n * self.n) - self.gamma * self.trProb[i]).dot(self.expR[i]) for i in range(4)]).reshape(4, self.n * self.n);
			print(L);
			self.V = np.max(L,0);
			self.Policy = np.argmax(L,0);

	def action(self, observation):
		
		return int(self.Policy[observation]);

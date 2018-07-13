"""
Dynamic Programming solutions for simple RL problems

MDP model is given to the class and optimal value function and policy are found out
"""
import numpy as np



class DP_Solns_RL:

	def __init__(self, MDP):

		self.MDP = MDP;		
		self.V = np.random.rand(self.MDP.nSta, 1); 
		self.Policy = np.random.randint(self.MDP.nAct, size = (self.MDP.nSta, 1));
		#self.Policy = np.array([[0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]]).T; #one of the solns for frozenLake-v0

	def ValueIteration(self, mode = "Iterative", cache = None):
	
	

		if mode == "Iterative":

			while True:
	
				L = np.array([self.MDP.expR[i] + self.MDP.gamma * self.MDP.trP[i].dot(self.V) for i in range(self.MDP.nAct) ]).reshape(self.MDP.nAct, self.MDP.nSta);
				Vn = np.max(L, 0).T[:, None];
				self.Policy = np.argmax(L, 0).T[:, None];
				print (np.linalg.norm(Vn - self.V));
				if np.linalg.norm(Vn - self.V) < self.MDP.eps * (1 - self.MDP.gamma)/(2 * self.MDP.gamma):
					self.V = Vn;
					break;
				else:
					self.V = Vn;

		if mode == "RTDP": # extreme GPI
			
			eps, obs = cache;
			L = np.array([self.MDP.expR[i][obs] + self.MDP.gamma * self.MDP.trP[i][obs][None, :].dot(self.V) for i in range(self.MDP.nAct) ]).reshape(self.MDP.nAct, 1);
			self.V[obs] = np.random.choice(np.array([np.random.choice(L[L != np.max(L, 0)], 1), np.max(L, 0)])[:, 0], 1, p = [eps, 1 - eps]);
			self.Policy[obs] = np.argmax(L == self.V[obs]);



	def PolicyIteration(self, mode = "Iterative", cache = None):
		
		if mode == "Iterative": 

			while True:

				#policy eval
				R = np.array([self.MDP.expR[int(i)][j] for (i,j) in zip(self.Policy,range(self.MDP.nSta))]);
				TrP = np.array([self.MDP.trP[int(i)][j, :] for (i,j) in zip(self.Policy,range(self.MDP.nSta))]);
				self.V = np.linalg.inv(np.identity(self.MDP.nSta) - self.MDP.gamma * TrP).dot(R);

				# policy imporvement
				L = np.array([self.MDP.expR[i] + self.MDP.gamma * self.MDP.trP[i].dot(self.V) for i in range(self.MDP.nAct) ]).reshape(self.MDP.nAct, self.MDP.nSta);
				Pin = np.argmax(L, 0).T[:, None];
				if (Pin == self.Policy).all():
					break;
				elif(np.linalg.norm(np.max(L, 0).T[:, None] - self.V) < self.MDP.eps * (1 - self.MDP.gamma)/(2 * self.MDP.gamma)):
					break;
				self.Policy = Pin;	


	def action(self, observation):
		
		return int(self.Policy[observation]);

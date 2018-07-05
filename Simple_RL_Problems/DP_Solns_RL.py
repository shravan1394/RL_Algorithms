"""
Dynamic Programming solutions for simple RL problems

MDP model is given to the class and optimal value function and policy are found out
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
		self.Policy = np.random.randint(4, size = (self.n * self.n, 1));
		#np.array([[1, 2, 1, 0, 0, 2, 1, 0, 3, 2, 1, 0, 2, 2, 1, 0]]).T; #one of the solns for frozenLake-v0

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



	def PolicyIteration(self, mode = "Iterative", cache = None):
		
		if mode == "Iterative": 


			while True:
				#policy eval

				R = np.array([self.expR[int(i)][j] for (i,j) in zip(self.Policy,range(self.n * self.n))]);

				TrP = np.array([self.trProb[int(i)][j, :] for (i,j) in zip(self.Policy,range(self.n * self.n))]);

				self.V = np.linalg.inv(np.identity(self.n * self.n) - self.gamma * TrP).dot(R) ;

				# policy imporvement
				L = np.array([self.expR[i] + self.gamma * self.trProb[i].dot(self.V) for i in range(4) ]).reshape(4,self.n * self.n);

				Pin = np.argmax(L,0).T[:,None];
				
				if (Pin == self.Policy).all():
					break;

				self.Policy = Pin;


		if mode == "GPI": # working on this!!!!!
			

			visitedStates = cache;

			#policy eval

			R = np.array([self.expR[int(i)][j] for (i,j) in zip(self.Policy,range(self.n * self.n))]);

			TrP = np.array([self.trProb[int(i)][j, :] for (i,j) in zip(self.Policy,range(self.n * self.n))]);

			self.V[visitedStates] = np.linalg.inv(np.identity(self.n * self.n) - self.gamma * TrP).dot(R)[visitedStates] ;


			# policy imporvement
			L = np.array([self.expR[i] + self.gamma * self.trProb[i].dot(self.V) for i in range(4) ]).reshape(4,self.n * self.n);

			self.Policy[visitedStates]  = np.argmax(L,0).T[:,None][visitedStates];
			



				
				

	def action(self, observation):
		
		return int(self.Policy[observation]);

"""

Generate MDP models for simple RL problems
"""
import numpy as np



class MDPModel:# working on it!!!



	def __init__(self, env, imR, gamma, eps):
		
		self.nSta = env.nS;
		self.nAct = env.nA;
		self.n = env.ncol;
		self.gamma = gamma;
		self.eps = eps; 
		self.imR = imR;
		self.trP = {0:self.TrPGen(env, 0),1:self.TrPGen(env, 1),2:self.TrPGen(env, 2),3:self.TrPGen(env, 3)};
		self.expR = {0:self.trP[0].dot(imR.T),1:self.trP[1].dot(imR.T),2:self.trP[2].dot(imR.T),3:self.trP[3].dot(imR.T)};


	def TrPGen(self, env, Dir):
	
		p = np.zeros((self.nSta, self.nSta));	
		for i in range(self.nSta):
			for j in range(len(env.P[i][Dir])):
				p[i, env.P[i][Dir][j][1]] += env.P[i][Dir][j][0];
		return p;
			

	

	


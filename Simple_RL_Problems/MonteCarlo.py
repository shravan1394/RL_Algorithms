"""
Solves simple RL problems

On-Policy Methods 
 1. Evaluating value function for a policy by sampling trajectories of it. Prbolem : control cant be done (i.e) policy cannot be improved
 2. Evaluate q instead  
"""

class MonteCarlo:
	
	def __init__(self, n, actions):
		
		self.V = np.random.rand(n * n, 1); 
		self.q = np.random.rand(n * n, actions); 
		self.Policy = np.random.randint(actions, size = (self.n * self.n, 1));


	def epsSoftPolicy():
		
		

	def action(self, observation):
		
		return int(self.Policy[observation]);

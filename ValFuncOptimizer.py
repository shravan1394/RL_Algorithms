import numpy as np

def nextAction(Q, cache = (0.1), policy = 'e-greedy'):

	problems, N = Q.shape;
	

	if policy == 'greedy':

		nextActions = np.apply_along_axis(np.argmax, 1, Q);
	
	if policy == 'e-greedy':

		# Masks for getting max Q in each problem
		maxMask = np.zeros(Q.shape);
		maxMask[np.arange(problems), np.apply_along_axis(np.argmax, 1, Q)] = 1;
	
		# Nextaction find: 1.) get a Q that is not the max assuming a uniform distribution.
		# 2.) choose max Q action and the uniform random Q action with a probability of 1-eps and eps resply
		notMaxDist = np.apply_along_axis(np.random.choice, 1, np.where(maxMask != 1)[1].reshape(problems, N - 1), 1); 
		distr = np.hstack((notMaxDist, np.where(maxMask == 1)[1][:,None]));

		eps = cache;
		nextActions = np.apply_along_axis(np.random.choice, 1, distr, p = [eps, 1 - eps]);

	if policy == 'softmax':

		tau = cache;
		prob = np.exp(Q / tau) / np.sum(np.exp(Q / tau), 1)[:,None];
		nextActions = np.array([np.random.choice(np.arange(N), p = prob[i,:]) for i in range(prob.shape[0])]);

	if policy == 'UCB1':
		
		i, Na = cache;
		confInterval = np.sqrt(2 * np.log(i) / (Na + 1));
		upperBound = Q + confInterval;
		nextActions = np.apply_along_axis(np.argmax, 1, upperBound);
	
	return nextActions;

def update(Q, Na, action, cache, rule = 'mean'):
	
	Arms = cache;
	problems, N = Q.shape;
	
	if rule == 'mean':

		# Update Q
		Q[np.arange(problems), action] += (1 / (Na[np.arange(problems), action] + 1)) * (Arms.pullArm(action = action) - Q[np.arange(problems), action]);
		
		# Update Na
		Na[np.arange(problems), action] += 1;



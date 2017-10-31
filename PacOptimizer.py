import numpy as np

def PacArm(eps, delta, cache, mode = 'naive'):
	
	if mode == 'naive':
		Arms = cache;
		N = Arms.N;
		l = int((2 / eps ** 2) * np.log(2 * N / delta));
		print("arm needs to be pulled " + str(l) + " times.");
		optimArm = np.argmax(Arms.pullArm(N,l).mean(1)); 

	if mode == 'median':
		Arms = cache;
		N, = mean.shape
		eps /= 4;
		delta /= 2;
		action = np.ones(N, dtype = np.bool8);
		while(Q.shape[0] != 1):
			l = int((2 / eps ** 2) * np.log(3 / delta));
			Q = (var[action][:, None] * X(N, l) + mean[action][:, None]).mean(1);
			action = np.where(mean[Q >= np.median(Q)])[0];
			eps *= 0.75;
			delta /= 2;
		#optimArm = np.where(
		
	return optimArm

import numpy as np

def PacArm(eps, delta, cache, mode = 'naive'):
	
	if mode == 'naive':
		Arms = cache;
		N = Arms.N;
		l = int((4 / eps ** 2) * np.log(2 * N / delta)/np.log(2));
		print l
		print("Each arm needs to be pulled " + str(l*N) + " times.");
		optimArm = np.argmax(Arms.pullArm(problems = l).mean(0)); 

	if mode == 'median':
		Arms = cache;
		N = Arms.N;
		eps /= 4;
		delta /= 2;
		L = 0;
		action = np.arange(N);
		while(N != 1):
			l = int((4 / eps ** 2) * np.log(3 / delta)/np.log(2));
			Q = Arms.pullArm(N, l, action).mean(0);
			action = action[(Q >= np.median(Q))];
			print(N)
			print(L)
			L += (l * N);	
			N = action.shape[0];									
			eps *= 0.75;
			delta /= 2;

		print("Total No. of arm pulls is" + str(L));
		print(L)
		optimArm = action[0];
		print (optimArm);
	return optimArm

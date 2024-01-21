import numpy as np

def DISUS(T_1, beta, N, bandit, tau=0.2, public_randomness=True):
	T_j = T_1
	t = 0
	curr_domain = bandit.domain
	domain_size = curr_domain.shape[1]
	reg = np.zeros(bandit.T)
	batch_ctr = 0
	comm_cost = np.zeros(bandit.T)

	y_j = np.zeros((T_j, N))
	sigma_vec = np.zeros(domain_size)
	mu_vec = np.zeros(domain_size)

	while True:

		samp_idxs = np.random.randint(low=0, high=domain_size, size=T_j*N)
		X_serv = curr_domain[:, samp_idxs] 
		start_idxs = np.arange(N)*T_j 
		for i in range(T_j):
			f_t = np.array([bandit.reward_func(X_serv[:, start_idxs[n] + i]) for n in range(N)])
			reg[t] = -np.sum(f_t)
			y_j[i, :] = f_t + np.random.normal(scale=bandit.noise_std, size=N)
			t += 1
			if t == bandit.T:
				break

		if t == bandit.T:
			break

		p_ind = 10*np.log(N*T_j)/(N*T_j)
		inducing_set_idxs = np.random.random(size=T_j*N) < p_ind
		Z = X_serv[:, inducing_set_idxs]
		n_inducing_pts = np.sum(inducing_set_idxs)

		comm_cost[t] = N*n_inducing_pts

		k_ZX = np.zeros((n_inducing_pts, T_j*N))
		for i in range(n_inducing_pts):
			x_i = np.reshape(Z[:, i], (bandit.dim, 1))
			x_diff = X_serv - x_i
			k_ZX[i, :] = bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0)))

		x_diff = 0
		for d in range(bandit.dim):
			x_samp = np.tile(Z[d], (n_inducing_pts, 1))
			x_diff += (x_samp - np.transpose(x_samp))**2

		K_ZZ = bandit.kernel(np.sqrt(x_diff))

		K_ZZ_inv = np.linalg.pinv(K_ZZ)
		K_inv = np.linalg.pinv((tau*K_ZZ + np.matmul(k_ZX, np.transpose(k_ZX))))

		bar_v = np.matmul(K_inv, np.matmul(k_ZX, np.matrix.flatten(y_j, order='F')))

		for i in range(domain_size):
			x_t = np.reshape(curr_domain[:, i], (bandit.dim, 1))
			x_diff = Z - x_t
			k_vec = bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0)))
			sigma_vec[i] = np.sqrt((1 - np.dot(k_vec, np.matmul(K_ZZ_inv, np.transpose(k_vec))))/tau + np.dot(k_vec, np.matmul(K_inv, np.transpose(k_vec))))
			mu_vec[i] = np.dot(k_vec, bar_v)

		if public_randomness:
			UCB = mu_vec + beta*sigma_vec
			LCB = mu_vec - beta*sigma_vec
			curr_domain = curr_domain[:, UCB >= np.max(LCB)]
			domain_size = curr_domain.shape[1]
		else:
			sigma_max = np.max(sigma_vec)
			mu_max = np.max(mu_vec)
			curr_domain = curr_domain[:, mu_vec >= mu_max - beta*sigma_max]
			domain_size = curr_domain.shape[1]

		T_j = int(np.ceil(np.sqrt(bandit.T*T_j)))

		y_j = np.zeros((T_j, N))
		sigma_vec = np.zeros(domain_size)
		mu_vec = np.zeros(domain_size)


	reg += N*bandit.max_reward
	return np.cumsum(reg), np.cumsum(comm_cost)













import numpy as np

def NKernelUCB(beta, N, bandit, tau=0.2):
	curr_domain = bandit.domain
	domain_size = curr_domain.shape[1]
	reg = np.zeros(bandit.T)
	comm_cost = np.zeros(bandit.T)


	for n in range(N):
		X = np.zeros((bandit.dim, bandit.T))
		y = np.zeros(bandit.T)
		sigma_vec = np.zeros(domain_size)
		mu_vec = np.zeros(domain_size)

		for t in range(bandit.T):

			if t == 0:
				samp_idx = np.random.randint(domain_size)
			else:
				UCB = mu_vec + beta*sigma_vec
				samp_idx = np.argmax(UCB)

			X[:, t] = bandit.domain[:, samp_idx]
			f_t = bandit.reward_func(X[:, t])
			reg[t] -= f_t
			y[t] =  f_t + np.random.normal(scale=tau)

			if t == 0:
				K_inv = np.array([[1/(1 + tau)]])
			else:
				x_diff = X[:, 0:t] - np.reshape(X[:, t], (bandit.dim, 1))
				k_vec = bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0)))
				b_K = np.array([np.matmul(k_vec, K_inv)])
				K_22 = 1/(1 + tau - np.dot(b_K, np.transpose(k_vec)))
				K_11 = K_inv + K_22*np.dot(np.transpose(b_K), b_K)
				K_21 = -K_22*b_K
				top_row = np.append(K_11, np.reshape(np.transpose(K_21), (t, 1)), axis=1)
				bottom_row = np.array([np.append(K_21, K_22)])
				K_inv = np.append(top_row, bottom_row, axis=0)

			theta = np.dot(K_inv, np.transpose(y[:(t+1)]))
			for i in range(domain_size):
				x_t = np.reshape(bandit.domain[:, i], (bandit.dim, 1))
				x_diff = X[:, 0:(t+1)] - x_t
				k_vec = bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0)))
				sigma_vec[i] = np.sqrt(1 - k_vec @ (K_inv @ np.transpose(k_vec)))
				mu_vec[i] = np.dot(k_vec, theta)

	reg += N*bandit.max_reward
	return np.cumsum(reg), np.cumsum(comm_cost)






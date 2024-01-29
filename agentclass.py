import numpy as np
import copy

class agent():
	def __init__(self, beta, N, bandit, to_approx=False, bar_q=4, tau=0.2):

		self.bandit = bandit

		self.N = N

		self.beta = beta

		self.tau = tau

		self.to_approx = to_approx

		self.X = np.zeros((bandit.dim, bandit.T*N))
		self.y = np.zeros(bandit.T*N)

		self.n_samp = 0

		self.K_inv = 0
		self.log_det_K = 0

		self.last_synchorize = 0

		self.domain_size = bandit.domain.shape[1]

		self.mu = np.zeros(self.domain_size)
		self.sigma = np.ones(self.domain_size)

		self.regret = np.zeros(bandit.T)
		self.comm_cost = np.zeros(bandit.T)

		self.local_time = 0

		#########

		self.approximating_set = 0
		self.n_approx_pts = 0
		# self.Z_global = 0
		# self.Z_local = 0
		self.Zy_local = 0
		self.ZT_Z_local = 0
		self.ZT_Z_inv_local = 0

		self.approx_sigma_sq_last_synchronize = np.ones(self.domain_size)
		self.sampled_idxs = np.zeros(bandit.T,  dtype=int)
		self.bar_q = bar_q




	def sample_next_point(self):

		if self.to_approx:
			self._sample_next_point_approx()
		else:
			self._sample_next_point_no_approx()

	def _sample_next_point_no_approx(self):

		if self.n_samp == 0:
			samp_idx = np.random.randint(self.domain_size)
		else:
			UCB = self.mu + self.beta*self.sigma
			samp_idx = np.argmax(UCB)

		self.X[:, self.n_samp] = self.bandit.domain[:, samp_idx]
		f_t = self.bandit.reward_func(self.X[:, self.n_samp])
		self.regret[self.local_time] = -f_t
		self.y[self.n_samp] =  f_t + np.random.normal(scale=self.tau)

		self.log_det_K += np.log(1 + (self.sigma[samp_idx]**2)/self.tau)

		if self.n_samp == 0:
			self.K_inv = np.array([[1/(1 + self.tau)]])
		else:
			x_diff = self.X[:, 0:self.n_samp] - np.reshape(self.X[:, self.n_samp], (self.bandit.dim, 1))
			k_vec = self.bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0)))
			b_K = np.array([np.matmul(k_vec, self.K_inv)])
			K_22 = 1/(1 + self.tau - np.dot(b_K, np.transpose(k_vec)))
			K_11 = self.K_inv + K_22*np.dot(np.transpose(b_K), b_K)
			K_21 = -K_22*b_K
			top_row = np.append(K_11, np.reshape(np.transpose(K_21), (self.n_samp, 1)), axis=1)
			bottom_row = np.array([np.append(K_21, K_22)])
			self.K_inv = np.append(top_row, bottom_row, axis=0)

		theta = np.dot(self.K_inv, np.transpose(self.y[:(self.n_samp+1)]))
		for i in range(self.domain_size):
			x_t = np.reshape(self.bandit.domain[:, i], (self.bandit.dim, 1))
			x_diff = self.X[:, 0:(self.n_samp+1)] - x_t
			k_vec = self.bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0)))
			self.sigma[i] = np.sqrt(1 - k_vec @ (self.K_inv @ np.transpose(k_vec)))
			self.mu[i] = np.dot(k_vec, theta)

		self.n_samp += 1
		self.local_time += 1


	def _sample_next_point_approx(self):

		if self.n_approx_pts == 0:
			samp_idx = np.random.randint(self.domain_size)
		else:
			UCB = self.mu + self.beta*self.sigma
			samp_idx = np.argmax(UCB)

		self.sampled_idxs[self.n_samp] = int(samp_idx)

		### n_samp and local_time are the same in this case

		self.X[:, self.n_samp] = self.bandit.domain[:, samp_idx]
		f_t = self.bandit.reward_func(self.X[:, self.n_samp])
		self.regret[self.local_time] = -f_t
		self.y[self.n_samp] =  f_t + np.random.normal(scale=self.tau)

		self.log_det_K += (self.sigma[samp_idx]**2)*self.tau

		if self.n_approx_pts > 0:
			x_diff = self.approximating_set[:, 0:self.n_approx_pts] - np.reshape(self.X[:, self.n_samp], (self.bandit.dim, 1))
			k_vec = self.bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0)))
			z_new = k_vec @ self.K_inv 
			self.Zy_local += z_new*self.y[self.n_samp]


			self.ZT_Z_local += np.outer(z_new, z_new)
			b = z_new @ self.ZT_Z_inv_local 
			self.ZT_Z_inv_local -= (np.outer(b, b))/(1 + z_new @ np.transpose(b))

			theta = self.ZT_Z_inv_local @ self.Zy_local
			K_approx_inv_local = self.ZT_Z_local @ self.ZT_Z_inv_local

			for i in range(self.bandit.domain_size):
				x_t = np.reshape(self.bandit.domain[:, i], (self.bandit.dim, 1))
				x_diff = self.approximating_set - x_t
				k_vec = self.bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0))) @ self.K_inv
				self.sigma[i] = np.sqrt((1 - k_vec @ (K_approx_inv_local @ np.transpose(k_vec)))/self.tau)  
				self.mu[i] = np.dot(k_vec, theta)

		self.n_samp += 1
		self.local_time += 1

	def upload_local_approximating_set(self):

		choose_idxs = (np.random.random(self.n_samp) - self.bar_q*self.approx_sigma_sq_last_synchronize[self.sampled_idxs[0:self.n_samp]]) < 0
		samp_idxs_local = self.sampled_idxs[0:self.n_samp]
		approx_set_idxs = samp_idxs_local[choose_idxs]
		self.comm_cost[self.n_samp-1] = approx_set_idxs.size*self.bandit.dim

		return approx_set_idxs

	def upload_new_ZTZ_Zy(self):

		ZTZ = np.zeros((self.n_approx_pts, self.n_approx_pts))
		Zy = np.zeros(self.n_approx_pts)
		for i in range(self.n_samp):
			x_i = np.reshape(self.X[:, i], (self.bandit.dim, 1))
			x_diff = self.approximating_set - x_i
			z_new = self.bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0))) @ self.K_inv
			Zy += z_new * self.y[i]
			ZTZ += np.transpose(z_new) @ z_new

		self.comm_cost[self.n_samp-1] = self.n_approx_pts*(self.n_approx_pts + 1)

		return ZTZ, Zy


	def upload_at_synchronize(self):

		X_new = self.X[:, self.last_synchorize:self.n_samp]
		y_new = self.y[self.last_synchorize:self.n_samp]

		self.last_synchorize += (self.n_samp - self.last_synchorize)*self.N
		self.n_samp = self.last_synchorize

		return X_new, y_new

	def download_at_synchronize(self, K_inv, log_det_K, X, y):

		self.X[:, 0:self.last_synchorize] = copy.deepcopy(X)
		self.y[0:self.last_synchorize] = copy.deepcopy(y)

		self.log_det_K = copy.deepcopy(log_det_K)
		self.K_inv = copy.deepcopy(K_inv)

		theta = np.dot(self.K_inv, np.transpose(self.y[:self.n_samp]))
		for i in range(self.domain_size):
			x_t = np.reshape(self.bandit.domain[:, i], (self.bandit.dim, 1))
			x_diff = self.X[:, 0:self.n_samp] - x_t
			k_vec = self.bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0)))
			self.sigma[i] = np.sqrt(1 - np.dot(k_vec, np.dot(self.K_inv, np.transpose(k_vec))))
			self.mu[i] = np.dot(k_vec, theta)




import numpy as np
from agentclass import *
import copy
from scipy.linalg import sqrtm

def ApproxDisKernelUCB(beta, N, bandit, D, bar_q, tau=0.2):
	log_det_K_global = 0
	t_last = 0
	regret = 0
	comm_cost = 0

	agents = [agent(beta=beta, N=N, bandit=bandit, tau=tau, to_approx=True, bar_q=bar_q) for _ in range(N)]

	for t in range(bandit.T):
		for n in range(N):
			agents[n].sample_next_point()

		trigger = False
		for n in range(N):
			condition = (agents[n].log_det_K - log_det_K_global)
			trigger = (trigger or (condition > D))

		if (trigger or (t == 10)):  # Added an additional condition to ensure that the first batch is small
			log_det_K_global = 0
			approximating_set_idxs = np.array([])
			for n in range(N):
				approximating_set_idxs = np.append(approximating_set_idxs, agents[n].upload_local_approximating_set())

			approximating_set_idxs = np.unique(approximating_set_idxs).astype(int)
			n_approx_pts = len(approximating_set_idxs)
			print(n_approx_pts)


			approximating_set = bandit.domain[:, approximating_set_idxs]

			x_diff = 0
			for d in range(bandit.dim):
				x_samp = np.tile(approximating_set[d, :], (n_approx_pts, 1))
				x_diff += (x_samp - np.transpose(x_samp))**2

			K = bandit.kernel(np.sqrt(x_diff))

			K_inv = sqrtm(np.linalg.pinv(K))

			for n in range(N):
				agents[n].approximating_set = approximating_set
				agents[n].n_approx_pts = n_approx_pts
				agents[n].K_inv = K_inv

			Zy_global = 0
			ZTZ_global = 0
			for n in range(N):
				ZTZ_local, Zy_local = agents[n].upload_new_ZTZ_Zy()
				ZTZ_global += ZTZ_local
				Zy_global += Zy_local

			ZTZ_inv_global = np.linalg.inv(ZTZ_global + tau*np.eye(n_approx_pts))
			sigma_sq_vec = np.zeros(bandit.domain_size)

			K_approx_inv = ZTZ_global @ ZTZ_inv_global

			for i in range(bandit.domain_size):
				x_t = np.reshape(bandit.domain[:, i], (bandit.dim, 1))
				x_diff = approximating_set - x_t
				k_vec = bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0))) @ K_inv
				sigma_sq_vec[i] = (1 - k_vec @ (K_approx_inv @ np.transpose(k_vec))/tau) 

			for n in range(N):
				agents[n].Zy_local = copy.deepcopy(Zy_global)
				agents[n].ZT_Z_local = copy.deepcopy(ZTZ_global)
				agents[n].ZT_Z_inv_local = copy.deepcopy(ZTZ_inv_global)
				agents[n].log_det_K = 0


			t_last = t


	for n in range(N):
		regret += agents[n].regret
		comm_cost += agents[n].comm_cost

	regret += N*bandit.max_reward
	return np.cumsum(regret), np.cumsum(comm_cost)



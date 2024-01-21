import numpy as np
from agentclass import *
import copy

def DisKernelUCB(beta, N, bandit, D, tau=0.2):
	log_det_K_global = 0
	K_inv_global = np.array([])
	X = np.zeros((bandit.dim, bandit.T*N))
	y = np.zeros(bandit.T*N)
	t_last = 0
	regret = 0
	comm_cost = 0

	agents = [agent(beta=beta, N=N, bandit=bandit, tau=tau) for _ in range(N)]

	for t in range(bandit.T):
		for n in range(N):
			agents[n].sample_next_point()

		trigger = False
		for n in range(N):
			condition = (t - t_last + 1)*(agents[n].log_det_K - log_det_K_global)
			trigger = (trigger or (condition > D))

		if trigger:
			n_new_samp = (t - t_last + 1)
			X_new = np.zeros((bandit.dim, n_new_samp*N))
			y_new = np.zeros(n_new_samp*N)
			for n in range(N):
				X_loc, y_loc = agents[n].upload_at_synchronize()
				X_new[:, n_new_samp*n:n_new_samp*(n+1)] = X_loc
				y_new[n_new_samp*n:n_new_samp*(n+1)] = y_loc
				agents[n].comm_cost[t] = n_new_samp*(bandit.dim + 1)
			# t_last += n_new_samp

			x_diff = 0
			for d in range(bandit.dim):
				x_samp = np.tile(X_new[d], (n_new_samp*N, 1))
				x_diff += (x_samp - np.transpose(x_samp))**2

			K_new = bandit.kernel(np.sqrt(x_diff))

			if K_inv_global.size == 0:
				K_inv_global = np.linalg.inv(K_new + tau*np.eye(n_new_samp*N))
				eig_vals = np.linalg.eigvals(K_new)
				log_det_K_global = np.sum(np.log(1+eig_vals/tau))
			else:
				K_new_old = np.zeros((n_new_samp*N, t_last*N))
				for i in range(n_new_samp*N):
					x_i = np.reshape(X_new[:, i], (bandit.dim, 1))
					x_diff = X[:, 0:t_last*N] - x_i
					K_new_old[i, :] = bandit.kernel(np.sqrt(np.sum(x_diff**2, axis=0)))

				SC_inv = np.linalg.inv(K_new + tau*np.eye(n_new_samp*N) - K_new_old @ (K_inv_global @ np.transpose(K_new_old)))
				K_21 = - SC_inv @ (K_new_old @ K_inv_global)
				bottom_row = np.append(K_21, SC_inv, axis=1)
				K_11 = K_inv_global - np.transpose(K_21) @ (K_new_old @ K_inv_global)
				top_row = np.append(K_11, np.transpose(K_21), axis=1)
				K_inv_global = np.append(top_row, bottom_row, axis=0)
				log_det_K_global -= (np.sum(np.log((np.linalg.eigvals(SC_inv)))) + np.log(tau)*(n_new_samp)*N)

			X[:, (t_last*N):(t_last*N + n_new_samp*N)] = X_new
			y[(t_last*N):(t_last*N + n_new_samp*N)] = y_new

			t_last += n_new_samp

			for n in range(N):
				agents[n].download_at_synchronize(K_inv=K_inv_global, log_det_K=log_det_K_global, X=X[:, 0:t_last*N], y=y[0:t_last*N])

	for n in range(N):
		regret += agents[n].regret
		comm_cost += agents[n].comm_cost

	regret += N*bandit.max_reward
	return np.cumsum(regret), np.cumsum(comm_cost)



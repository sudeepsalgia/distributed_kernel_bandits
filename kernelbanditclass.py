import numpy as np
import copy

class KernelBandit():
	def __init__(self, T, reward_func, kernel_params, dim, domain_size, noise_std=0.2, max_reward = -np.Inf, reg_domain=False):

		# Set seed to ensure that the original domain is the same across iterations
		# self.seed = seed

		# Set the time horizon
		self.T = T

		self.reward_func = reward_func

		self.max_reward = max_reward

		self.kernel_params = kernel_params

		self.dim = dim

		self.domain_size = domain_size

		self.noise_std = noise_std

		self.reg_domain = reg_domain

		self.sqrt3 = np.sqrt(3)
		self.sqrt5 = np.sqrt(5)

	def generate_domain(self):

		# np.random.seed(self._seed)
		if self.reg_domain:
			x0 = np.linspace(0,1,10,endpoint=False)
			self._domain = np.reshape(np.array(np.meshgrid(x0,x0,x0,x0)), (4, 10000))
		else:
			self._domain = np.random.rand(self.dim, self.domain_size)

		if np.isinf(self.max_reward):
			self.max_reward = max([self.reward_func(x) for x in np.transpose(self._domain)])

		# x = np.linspace(0,1,30)
		# X, Y = np.meshgrid(x,x)
		# self._domain = np.array([X.flatten(), Y.flatten()])

		# return domain

	def reset_domain(self):

		self.domain = copy.deepcopy(self._domain)

	def kernel(self, r):
		l = self.kernel_params[1]
		if self.kernel_params[0] == np.Inf:
			k = np.exp(-(r**2)/(2*(l**2)))
		elif self.kernel_params == 1.5:
			k = (1 + self.sqrt3*r/l)*np.exp(-self.sqrt3*r/l)
		elif self.kernel_params == 2.5:
			k = (1 + self.sqrt5*r/l + 5*(r**2)/(l**2))*np.exp(-self.sqrt5*r/l)

		return k

	def generate_tree(self):
		
		X = copy.deepcopy(self._domain)

		dim, n_pts = X.shape

		nodes = {}
		children = {}
		
		nodes[0] = X
		node_ctr = 1

		curr_nodes = [(X, 0, 0)]

		const_vec = np.array([[2**i for i in range(dim)]])

		while curr_nodes:

			node, depth, node_ctr_local = curr_nodes.pop(0)
			depth += 1
			idxs = np.squeeze(np.dot(const_vec, np.floor(node*(2**depth))%2).astype(int))
			# print(idxs)

			idx_map = {}
			for i in range(len(idxs)):
				if idxs[i] in idx_map:
					idx_map[idxs[i]].append(node[:, i])
				else:
					idx_map[idxs[i]] = [node[:, i]]

			for n in idx_map.values():
				n = np.transpose(np.array(n))

				nodes[node_ctr] = n

				if node_ctr_local in children:
					children[node_ctr_local].append(node_ctr)
				else:
					children[node_ctr_local] = [node_ctr]

				if n.shape[1] > 1:
					curr_nodes.append((n, depth, node_ctr))

				node_ctr += 1

		# print(depth)

		return nodes, children


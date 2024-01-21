import numpy as np
import matplotlib.pyplot as plt
import time
from kernelbanditclass import *
from DISUS import *
from DisKernelUCB import *
from ApproxDisKernelUCB import *
from NKernelUCB import *

d = 10
theta_star = np.random.normal(d)
theta_star = theta_star/np.linalg.norm(theta_star)

def polynomial_ip(x):

	z = np.dot(x, theta_star)

	return np.cos(z**3 - 3*(z**2) + 3*z + 3)



T = 100
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

bandit = KernelBandit(T=T, reward_func=polynomial_ip, kernel_params=[np.Inf, 1], dim=d, domain_size=2000, noise_std=0.2, cube_domain=False)
bandit.generate_domain()

# Set parameters
T_1 = 5
beta = 1
N = 10
D_DisKernelUCB = T/(N*np.log(N*T))
D_ApproxDisKernelUCB = 1/N
bar_q = 4

n_loops = 2
reg_ApproxDisKernelUCB = np.zeros((n_loops, T))
reg_DISUS = np.zeros((n_loops, T))
reg_DisKernelUCB = np.zeros((n_loops, T))
reg_NKernelUCB = np.zeros((n_loops, T))

comm_cost_ApproxDisKernelUCB = np.zeros((n_loops, T))
comm_cost_DISUS = np.zeros((n_loops, T))
comm_cost_DisKernelUCB = np.zeros((n_loops, T))
comm_cost_NKernelUCB = np.zeros((n_loops, T))



green_color = (0.4660,0.7640,0.1880)
purple_color = (0.4940,0.1840,0.5560)
orange_color = 'orange'
red_color = 'red'

# Runscript

for i in range(n_loops):

	bandit.reset_domain()

	start = time.time()
	reg_ApproxDisKernelUCB[i, :], comm_cost_ApproxDisKernelUCB[i, :] = ApproxDisKernelUCB(beta=beta, N=N, bandit=bandit, D=D_ApproxDisKernelUCB, bar_q=bar_q) 
	end = time.time()
	# time_BPE[i] = (end - start)

	bandit.reset_domain()

	start = time.time()	
	reg_DISUS[i, :], comm_cost_DISUS[i, :] = DISUS(T_1=T_1, beta=beta, N=N, bandit=bandit,)
	end = time.time()
	# time_REDS[i] = (end - start)

	bandit.reset_domain()

	start = time.time()
	reg_DisKernelUCB[i, :], comm_cost_DisKernelUCB[i, :] = DisKernelUCB(beta=beta, N=N, bandit=bandit, D=D_DisKernelUCB) 
	end = time.time()
	# time_GP_ThreDS[i] = (end - start)

	bandit.reset_domain()

	start = time.time()
	reg_NKernelUCB[i, :], comm_cost_NKernelUCB[i, :] = NKernelUCB(beta=beta, N=N, bandit=bandit) 
	end = time.time()
	# time_BPE[i] = (end - start)

	bandit.reset_domain()


fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1)

time = np.arange(T)

regrets = [reg_NKernelUCB, reg_DisKernelUCB, reg_ApproxDisKernelUCB, reg_DISUS]
comm_costs = [comm_cost_NKernelUCB, comm_cost_DisKernelUCB, comm_cost_ApproxDisKernelUCB, comm_cost_DISUS]

labels = ['NKernelUCB', 'DisKernelUCB', 'ApproxDisKernelUCB', 'DISUS']
colors = [purple_color, orange_color, red_color, green_color]

for i in range(4):
	mean_regrets = np.mean(regrets[i], axis=0)
	std_regrets = np.std(regrets[i], axis=0) 
	ax.plot(time, mean_regrets, label=labels[i], color=colors[i])
	ax.fill_between(time, mean_regrets - std_regrets, mean_regrets + std_regrets, alpha=0.15, color=colors[i])
	# print(np.mean(times[i]), np.std(times[i]))

	mean_comm_costs = np.mean(comm_costs[i], axis=0)
	std_comm_costs = np.std(comm_costs[i], axis=0)
	print(mean_comm_costs[-1])



ax.set_ylabel('Regret', fontsize=15)
ax.set_xlabel('Time', fontsize=15)

plt.legend(loc='upper left', prop={'size': 16})
# plt.tight_layout()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Regret vs Time', fontsize=20)
plt.show()



from matplotlib import pyplot as plt
import numpy as np

n_episodes, n_runs = 300, 25
t = np.arange(n_episodes)

# an (n_episodes x n_runs) array of random walk steps
S1 = 0.002 + 0.01*np.random.randn(n_runs, n_episodes)
# S2 = 0.004 + 0.02*np.random.randn(n_episodes, n_runs)

# an (n_episodes x n_runs) array of random walker positions
# X1 = S1.cumsum(axis=0)
X1 = S1
# X2 = S2.cumsum(axis=0)

print(X1.shape)


# n_episodes length arrays empirical means and standard deviations of both
# populations over time
mu1 = X1.mean(axis=0)
sigma1 = X1.std(axis=0)
# mu2 = X2.mean(axis=1)
# sigma2 = X2.std(axis=1)

# plot it!
fig, ax = plt.subplots(1)
# mean line
ax.plot(t, mu1, lw=2, label='mean population 1', color='blue')
# ax.plot(t, mu2, lw=2, label='mean population 2', color='yellow')
# std line
ax.fill_between(t, mu1+sigma1, mu1-sigma1, facecolor='blue', alpha=0.2)
# ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.2)
ax.set_title(r'random walkers empirical $\mu$ and $\pm \sigma$ interval')
ax.legend(loc='upper left')
ax.set_xlabel('num steps')
ax.set_ylabel('position')
ax.grid()

# ax.plot(X1[0])

plt.show()

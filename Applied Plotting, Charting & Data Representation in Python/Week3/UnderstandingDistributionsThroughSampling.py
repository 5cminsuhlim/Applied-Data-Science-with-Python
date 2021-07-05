#GIVEN CODE
import matplotlib.pyplot as plt
import numpy as np

# generate 4 random variables from the random, gamma, exponential, and uniform distributions
x1 = np.random.normal(-2.5, 1, 10000)
x2 = np.random.gamma(2, 1.5, 10000)
x3 = np.random.exponential(2, 10000)+7
x4 = np.random.uniform(14,20, 10000)

# plot the histograms
plt.figure(figsize=(9,3))
plt.hist(x1, normed=True, bins=20, alpha=0.5)
plt.hist(x2, normed=True, bins=20, alpha=0.5)
plt.hist(x3, normed=True, bins=20, alpha=0.5)
plt.hist(x4, normed=True, bins=20, alpha=0.5);
plt.axis([-7,21,0,0.6])

plt.text(x1.mean()-1.5, 0.5, 'x1\nNormal')
plt.text(x2.mean()-1.5, 0.5, 'x2\nGamma')
plt.text(x3.mean()-1.5, 0.5, 'x3\nExponential')
plt.text(x4.mean()-1.5, 0.5, 'x4\nUniform')


#MY CODE
import matplotlib.animation as animation

plots = [x1, x2, x3, x4]
titles = ['Normal', 'Gamma', 'Exponential', 'Uniform']

d1 = [-7.5, 2.5, 0, 1]
d2 = [0, 10, 0, 1]
d3 = [7.5, 17.5, 0, 1]
d4 = [12.5, 22.5, 0, 1]
dims = [d1, d2, d3, d4]

bins = 20
n = 1000

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True)
axes = [ax1, ax2, ax3, ax4]

legends = [0.5, 8, 15.5, 20.5]

def update(curr):
    if curr == n:
        a.event_source.stop()
    for i in range(len(axes)):
        axes[i].cla()
        axes[i].hist(plots[i][:curr], normed = True, bins = bins)
        axes[i].axis(dims[i])
        axes[i].set_title(titles[i])
        axes[i].set_ylabel('Frequency')
        axes[i].set_xlabel('Value')
        axes[i].annotate('n = {}'.format(curr), [legends[i], 1])
    plt.tight_layout()

a = animation.FuncAnimation(fig, update, interval = 100)

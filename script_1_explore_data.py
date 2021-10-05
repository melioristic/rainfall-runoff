from RR.io import read_data
import matplotlib.pylab as plt
import numpy as np

Xd, Y, elp, flp, slp, area = read_data()

# Explore - Plot the data 1 year

fig = plt.figure(constrained_layout = True, figsize = (16,8))

gs = fig.add_gridspec(4,3)

ax1 = fig.add_subplot(gs[0,:2])
ax2 = fig.add_subplot(gs[1,:2])
ax3 = fig.add_subplot(gs[2,:2])
ax4 = fig.add_subplot(gs[3,:2])

x_axis = np.arange(365)

ax1.plot(Xd[:365,2], label = 'Solar Radiation')
ax2.plot(Xd[:365,1], label = 'Mean Temperature')
ax3.bar(x_axis, Xd[:365,0], label = 'Precipitation')
ax1.legend()
ax2.legend()
ax3.legend()

ax4.plot(Y[:365], label = 'Runoff')

plt.savefig('explore.png')
# TODO: 
# - sample a function for true state instead of data
# - sample a Gaussian noise distribution on your own
# - Add visualization of the Kalman gain to show how it moves
# - more scripts for vector formulation and sensor fusion
# - better notes and understanding of the prediction error and covariance and stuff

import matplotlib.pyplot as plt
import numpy as np

# time steps
k = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# truth for x_k = 0.75*x_(k-1) with X_0 = 1000, note a = 0.75
x_k = [1000, 750, 563, 422, 316, 237, 178, 133, 100, 75]
a = 0.75

# observations from a noisy sensor (we'll want to improve this with generated gaussian noise)
z_k = [890, 624, 644, 311, 156, 362, 145, 103, -7, 212]

# arbirary but strictly nonzero start for the prediction error
p_k = 1

# initial state estimate, setting it equal to the first observation
x_hat_k = z_k[0]

# the variance of the sensor noise
r = 200

# list to populate our estimates with
x_hats = []

### Kalman Filter ###
for z in z_k:
	# prediction phase
	x_hat_k = a * x_hat_k # system model
	p_k = a * p_k * a # prediction error

	# update phase
	g_k = p_k / (p_k + r) # kalman gain
	x_hat_k = x_hat_k + g_k * (z - x_hat_k) # state estimate
	p_k = (1-g_k) * p_k # prediction error update

	# added our estimate at each step to the list
	x_hats.append(x_hat_k)

fig, ax = plt.subplots()
ax.plot(k, x_k, label="truth")
ax.plot(k, z_k, label="measurements")
ax.plot(k, x_hats, label="Kalman filtered")

ax.legend()

ax.set(xlabel='timesteps k', ylabel='value', title='Kalman Filtering Noisy Data')

ax.grid()

plt.show()

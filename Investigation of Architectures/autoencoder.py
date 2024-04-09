#!/usr/bin/env python
# coding: utf-8

# # Generate synthetic data from a discrete spectrum dynamical system with a known Koopman invariant subspace

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

from dynamics import SlowManifold

# define system parameters

mu = -0.05
lam = -1

# create instance of SlowManifold class

system = SlowManifold(mu, lam)

# define initial conditions

x0 = np.linspace(-0.5, 0.5, 10000)
y0 = np.linspace(-0.5, 0.5, 10000)

# define time span

t_start = 0
t_end = 100
num_points = 1000

# solve the system of differential equations

t, x, y = system.solve(x0, y0, t_start, t_end, num_points)

# plot the results

plt.plot(x, x**2, 'r--', label='Slow manifold')
plt.plot(x, y, label='Nonlinear system')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Space Plot of Nonlinear Dynamical System with Slow Manifold')
plt.legend()
plt.grid(True)
plt.show()


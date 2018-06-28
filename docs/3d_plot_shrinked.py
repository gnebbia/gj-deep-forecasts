import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def fun(x, y):
  return ((((x-8)*300)**2 - ((y-8)*300)**2 + 2*((x)*300)*((y)*300)**2 + 1)/10**5) + 540 

def fun_bak(x, y):
  return x**2 - y**2 + 2*x*y**2 + 1 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#x = y = np.arange(-8.0, 8.0, 0.05)
x = y = np.arange(-1.0, 1.0, 0.001)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
print(Z)
print("The minimum value for the loss function is: " + str(Z.min()))
print("The maximmum value for the loss function is: " + str(Z.max()))

ax.plot_surface(X, Y, Z)

ax.set_xlabel('b-a')
ax.set_ylabel('sigma')
ax.set_zlabel('Loss')
fig.savefig('loss_function_plot.png')

plt.show()

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

x_shifting = 8
output_shrinking = 172 
z_shifting = 101.769
y_shifting = 10**5

#def fun(x, y):
#  return ((((x-8)*172)**2 - ((y-8)*172)**2 + 2*((x)*172)*((y)*172)**2 + 1)/10**5) + 101.769

def fun(x, y):
  return x**2 - y**2 - 2*x*y**2 + 1 

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

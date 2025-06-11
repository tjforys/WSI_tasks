from autograd import grad
from cec2017.functions import f1, f2, f3
import numpy as np
import matplotlib.pyplot as plt


def booth(point):
    x = point[0]
    y = point[1]
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


UPPER_BOUND = 100
DIMENSIONALITY = 10

x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY)
beta = 1e-10

MAX_X = 100
PLOT_STEP = 0.1

x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
X, Y = np.meshgrid(x_arr, y_arr)
Z = np.empty(X.shape)


q = f3
gradient = grad(q)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = q(np.array([X[i, j], Y[i, j]]))
plt.contour(X, Y, Z, 20)


i=0

while True:
    i += 1
    vector = gradient(x)

    prev_x = np.copy(x)
    x -= beta*vector

    x = np.array([UPPER_BOUND if a > UPPER_BOUND else a for a in x])
    x = np.array([-1*UPPER_BOUND if a < (-1*UPPER_BOUND) else a for a in x])

    plt.arrow(prev_x[0], prev_x[1], x[0]-prev_x[0], x[1]-prev_x[1], length_includes_head=True,
        head_width=0.08, head_length=0.00002)

    if all(abs(part) < 1e-6 for part in vector) or i > 1000:
        
        print(q(x))
        plt.plot(x[0], x[1], "ro") 
        break
plt.savefig("booth1.jpg")
plt.show()
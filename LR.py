import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]])

one = np.ones((x.shape[0], 1))
xbar = np.concatenate((one, x), axis = 1)
A = np.dot(xbar.T, xbar)
B = np.dot(y, xbar)
C = np.dot(B, np.linalg.pinv(A))

b = C[0][0]
w = C[0][1]

print(w)
print(b)

x0 = np.array([145, 183])
y0 = b + w*x0

# Drawing the fitting line
plt.plot(x.T, y, 'ro')     # data
plt.plot(x0, y0)
plt.show()

reg = linear_model.LinearRegression();
reg.fit(x.T, y)

regr.coef_

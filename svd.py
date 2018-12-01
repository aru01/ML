from numpy import array
from scipy.linalg import svd
from numpy import dot
from numpy import diag
from numpy import zeros
from math import sqrt

A = array([[0, 1, 1], [sqrt(2), 2, 0], [0, 1, 1]])
print(A)

U, s, VT = svd(A)
print(U)
print('')
print(s)
print('')
print(VT)

Sigma = diag(s)

B = U.dot(Sigma.dot(VT))
print(B)

//i'm use the tutoriial https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
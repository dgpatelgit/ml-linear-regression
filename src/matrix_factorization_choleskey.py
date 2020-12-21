# Cholesky decomposition
from numpy import array
from numpy.linalg import cholesky
# define a 3x3 matrix
A = array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
print(A)
# Cholesky decomposition
L = cholesky(A)
print(L)
# reconstruct
B = L.dot(L.T)
print(B)
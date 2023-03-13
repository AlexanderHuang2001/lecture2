

import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    
    for k in range(n-1):
        for i in range(k+1, n):
            factor = A[i,k]/A[k,k]
            for j in range(k+1, n):
                A[i,j] -= factor * A[k,j]
            b[i] -= factor * b[k]
    
    # Back substitution
    x = np.zeros(n)
    x[n-1] = b[n-1]/A[n-1,n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:]))/A[i,i]
    
    return x


A = np.array([[1,2], [0,1]])
b = np.array([5,2])

x = gaussian_elimination(A, b)

print(x)

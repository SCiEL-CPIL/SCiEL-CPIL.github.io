   import numpy as np

   def construct():
       n, m, p = 2, 5, 6
       # Standard decomposition
       # U: (n*m, rank)
       # V: (m*p, rank)
       # W: (p*n, rank)
       
       rank = n * m * p
       U = np.zeros((n * m, rank))
       V = np.zeros((m * p, rank))
       W = np.zeros((p * n, rank))
       
       idx = 0
       for i in range(n):
           for k in range(m):
               for j in range(p):
                   # C[i, j] += A[i, k] * B[k, j]
                   # U[ (i*m + k), idx ] = 1
                   # V[ (k*p + j), idx ] = 1
                   # W[ (j*n + i), idx ] = 1
                   
                   U[i * m + k, idx] = 1
                   V[k * p + j, idx] = 1
                   W[j * n + i, idx] = 1
                   idx += 1
                   
       return {'U': U, 'V': V, 'W': W}
   
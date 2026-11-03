import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120

def construct():
       n, m, p = 3, 3, 6
       rank = n * m * p  # 54
       
       # Vectorized construction of Schoolbook Decomposition
       # U[i*m+j, r] = 1 if r == i*m*p + j*p + k ... 
       # Actually, let's map r to (i, j, k)
       
       # r goes from 0 to 53
       # i = r // (m*p)
       # j = (r // p) % m
       # k = r % p
       
       U = np.zeros((n * m, rank))
       V = np.zeros((m * p, rank))
       W = np.zeros((p * n, rank))
       
       for r in range(rank):
           i = r // (m * p)
           j = (r // p) % m
           k = r % p
           
           U[i * m + j, r] = 1.0
           V[j * p + k, r] = 1.0
           W[k * n + i, r] = 1.0
       
       return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

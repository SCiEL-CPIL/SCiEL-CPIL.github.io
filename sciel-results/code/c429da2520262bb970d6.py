import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0  # wall-clock limit

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook decomposition
    # C = A * B
    # A is n x m, B is m x p, C is n x p
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # Tensor T[i, j, k] = A[i, j] * B[j, k] * C[k, i] (using trace convention W[k*n+i, r])
    # Actually, the problem states W[k*n+i, r].
    # The tensor decomposition is:
    # T = sum_r U[:, r] * V[:, r] * W[:, r]
    # where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i, i'} delta_{j, j'} delta_{k, k'} ?
    # No, it's the matrix multiplication tensor.
    # T_{i, j, k} = 1 if the indices match in the multiplication sense.
    # The standard tensor for matrix multiplication M_{n, m, p} has entries:
    # T_{(i, j), (j, k), (k, i)} = 1 for all i, j, k.
    # Here, U is indexed by (i, j), V by (j, k), W by (k, i).
    # The rank is n*m*p.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = (i * m + j) * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

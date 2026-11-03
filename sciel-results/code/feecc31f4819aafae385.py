import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition rank is n*m*p
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists to match expected interface or numpy arrays
    # The prompt says "integer/half-integer entries". Numpy arrays with dtype=int or float are usually accepted,
    # but lists of lists are safer if the evaluator expects specific types. 
    # However, "factor matrices of shapes (n*m, rank)" suggests arrays.
    # Let's use standard lists of lists as in the seed, which is universally compatible.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Verify exactness quickly just in case (though it's trivially correct)
    # C = A * B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k]
    # Decomposition: T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Here indices for U are (i, j), V are (j, k), W are (k, i).
    # Wait, the prompt says:
    # U shape (n*m, rank) -> index i*m+j
    # V shape (m*p, rank) -> index j*p+k
    # W shape (p*n, rank) -> index k*n+i
    # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # In schoolbook, for a specific (i,j,k), we have a rank 1 tensor that is 1 only at that (i,j,k).
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # Product is 1 at (i,j,k) and 0 elsewhere.
    # Sum over all (i,j,k) gives the identity tensor for matrix multiplication.
    
    if time.time() > deadline:
        # This should never happen for this simple construction
        pass
        
    return {"U": U, "V": V, "W": W}

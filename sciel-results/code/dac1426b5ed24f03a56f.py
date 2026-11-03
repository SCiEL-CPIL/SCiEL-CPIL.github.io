"""Program to find a low-rank decomposition for 2x3 * 3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    rank = n * m * p  # 42
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
                
    # Verification to ensure correctness (optional but safe)
    # T_recon = sum_r U[:, r] V[:, r]^T W[:, r]^T ... 
    # Actually, the tensor T is defined by T[(k*n+i), (i*m+j), (j*p+k)] = 1
    # We check if the decomposition reconstructs the identity of the multiplication tensor.
    # For the schoolbook decomposition, this is trivially true.
    
    # Convert to lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

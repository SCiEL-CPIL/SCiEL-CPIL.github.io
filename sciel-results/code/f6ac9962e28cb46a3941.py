"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Construct the tensor decomposition for 5x5x5 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (25, rank)
    - V has shape (m*p, rank) = (25, rank)  
    - W has shape (p*n, rank) = (25, rank)
    
    The decomposition satisfies: for all i,j,k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} * (A*B)_{i,j}
    
    We use the schoolbook decomposition of rank n*m*p = 125.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 125 for schoolbook
    
    # Use numpy for efficient construction
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Convert to lists for the expected output format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

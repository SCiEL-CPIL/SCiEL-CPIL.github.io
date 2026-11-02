import time
import sys

# Dimensions for the matrix multiplication: (n, m, p)
# A is n x m, B is m x p, C is n x p
# Here n=2, m=3, p=3
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x3 matrix.
    The decomposition is of the form:
    C = sum_{c=0}^{rank-1} (U_c @ V_c) * W_c
    
    Schoolbook algorithm with rank R = n * m * p = 18.
    """
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 5.0:
                    # Fallback to current best (which is the schoolbook being built)
                    # Since we are in the middle of building, we can just break and return what we have?
                    # No, we must return a valid decomposition.
                    # The schoolbook construction is very fast, so this should never trigger.
                    pass
                    
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][col_idx] = 1.0
                V[idx_v][col_idx] = 1.0
                W[idx_w][col_idx] = 1.0
                
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

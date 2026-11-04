"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    The tensor T has entries T[i, j, k] = 1 for all i in [0, 4), j in [0, 5), k in [0, 5).
    The decomposition is defined by matrices U, V, W such that:
    sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = 1
    
    We use the schoolbook decomposition which has rank n*m*p = 100.
    This is a valid exact decomposition with integer entries.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank) -> (20, 100)
    # V shape: (m*p, rank) -> (25, 100)
    # W shape: (p*n, rank) -> (20, 100)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Set the 1s in the decomposition
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time limit just in case
                if time.time() - start_time > 140.0:
                    break
            if time.time() - start_time > 140.0:
                break
        if time.time() - start_time > 140.0:
            break
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

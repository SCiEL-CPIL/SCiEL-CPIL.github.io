"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
_start = time.time()

def construct():
    n, m, p = DIMENSIONS
    # Use rank-36: standard schoolbook decomposition for all 9 columns
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # (4, 36)
    V = [[0] * rank for _ in range(m * p)]  # (18, 36)
    W = [[0] * rank for _ in range(p * n)]  # (18, 36)
    
    col = 0
    for k in range(p):
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][col] += 1
        V[0*p+k][col] += 1
        W[k*n+0][col] += 1
        col += 1
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[1][col] += 1
        V[1*p+k][col] += 1
        W[k*n+0][col] += 1
        col += 1
        
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[2][col] += 1
        V[0*p+k][col] += 1
        W[k*n+1][col] += 1
        col += 1
        
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[3][col] += 1
        V[1*p+k][col] += 1
        W[k*n+1][col] += 1
        col += 1
        
    assert col == rank, f"Expected {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

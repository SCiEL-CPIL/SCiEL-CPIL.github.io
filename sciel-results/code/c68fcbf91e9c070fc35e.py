"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Constructs a tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Uses the block structure: 3 independent 5x5 matrix multiplications.
    Rank is 3 * (5*5) = 75.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    # Rank is n * m * p for schoolbook
    rank = n * m * p
    
    # Initialize matrices
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over the 3 blocks (rows of A)
    for i in range(n):
        # Iterate over the 5x5 multiplication for this block
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (Trace convention: W[k*n+i, r])
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                
                col += 1
                
                # Time check (unlikely to trigger for small matrices, but good practice)
                if time.time() > deadline:
                    # Return partial result? No, schoolbook is fast.
                    # If we were searching, we would return best found.
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

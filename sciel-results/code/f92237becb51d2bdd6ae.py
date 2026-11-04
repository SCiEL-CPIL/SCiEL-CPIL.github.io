"""Program for matrix multiplication tensor decomposition (4x5 * 5x6)."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    Returns dict with keys 'U', 'V', 'W' where:
      U has shape (n*m, rank) = (20, rank)
      V has shape (m*p, rank) = (30, rank)
      W has shape (p*n, rank) = (24, rank)
    
    The decomposition satisfies:
      (A @ B)[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for all i, k, summed over j.
    """
    n, m, p = DIMENSIONS
    
    import random
    import numpy as np
    
    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Try to find a decomposition with reduced rank
    # Use structured random search with small integer entries
    for rank in range(50, 120):
        for trial in range(200):
            random.seed(trial * 7919 + rank * 104729)
            
            # Initialize with small integers
            U = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Find rank-minimal exact tensor decomposition for 2x3x6 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank decomposition for 2x3x6 matrix multiplication.
    
    For 2x3x6: n=2, m=3, p=6
    U: (n*m, rank) = (6, rank)
    V: (m*p, rank) = (18, rank)  
    W: (p*n, rank) = (12, rank)
    
    Strategy: Use two independent 2x3x3 schoolbook decompositions (rank 9 each)
    giving rank 18. Then verify correctness.
    """
    n, m, p = 2, 3, 6
    
    # Rank 18: two copies of 2x3x3 schoolbook
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 6 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 12 x 18
    
    # First block: B columns 0-2, C columns 0-2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                r = i * 9 + j * 3 + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    # Second block: B columns 3-5, C columns 3-5
    for i in range(n):
        for j in range(m):
            for k in range(3):
                r = 9 + i * 9 + j * 3 + k
                U[i * m + j][r] = 1
                V[j * p + (k + 3)][r] = 1
                W[(k + 3) * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

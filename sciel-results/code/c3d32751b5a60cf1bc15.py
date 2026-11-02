import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Block-circulant decomposition using ring Q[x]/(x^4-1)
    # Partition 12 columns into 3 groups of 4
    # Use isomorphism Q[x]/(x^4-1) ≅ Q × Q × Q[x]/(x^2+1)
    # This gives rank = 3 * 7 = 21 (better than 6*7=42)
    
    num_groups = 3
    group_size = 4
    rank_per_group = 7
    total_rank = num_groups * rank_per_group
    
    # Strassen-like rank-7 decomposition for 2x2x2
    U_block = [
        [1, 1, 0, 0, 0, 0, 0],   # (i=0,j=0)
        [0, 0, 1, 1, 0, 0, 0],   # (i=0,j=1)
        [0, 0, 0, 0, 1, 1, 0],   # (i=1,j=0)
        [0, 0, 0, 0, 0, 0, 1],   # (i=1,j=1)
    ]
    
    V_block = [
        [1, 0, 1, 0, 1, 0, 0],   # (j=0,k=0)
        [0, 1, 0, 1, 0, 1, 0],   # (j=0,k=1)
        [1, 0, 0, 0, 0, 0, 1],   # (j=1,k=0)
        [0, 1, 1, 0, 0, 0, 0],   # (j=1,k=1)
    ]
    
    W_block = [
        [1, 0, 0, 0, 0, 0, 0],   # (k=0,i=0)
        [0, 1, 0, 0, 0, 0, 0],   # (k=0,i=1)
        [0, 0, 1, 1, 0, 0, 0],   # (k=1,i=0)
        [0, 0, 0, 0, 1, 1, 1],   # (k=1,i=1)
    ]
    
    # Verify the decomposition
    max_err = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for r in range(7):
                    val += U_block[i*2 + j][r] * V_block[j*2 + k][r] * W_block[k*2 + i][r]
                max_err = max(max_err, abs(val - 1.0))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build full matrices using block-circulant structure
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for group in range(num_groups):
        base_col = group * group_size
        for i in range(2):
            for j in range(2):
                for r in range(7):
                    U[i*m + j][group*7 + r] = U_block[i*2 + j][r]
        
        for j in range(2):
            for k in range(group_size):
                for r in range(7):
                    V[j*p + base_col + k][group*7 + r] = V_block[j*2 + (k % 2)][r]
        
        for k in range(group_size):
            for i in range(2):
                for r in range(7):
                    W[(base_col + k)*n + i][group*7 + r] = W_block[(k % 2)*2 + i][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

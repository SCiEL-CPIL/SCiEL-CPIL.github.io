"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    # Rank 6 decomposition: one term per (j, k) pair
    # r = j * p + k for j in [0, m), k in [0, p)
    rank = m * p  # 3 * 3 = 9? No, we want (j,k) pairs so m*p = 9
    # Wait, j has m=3 values, k has p=3 values, so m*p = 9 pairs
    # But we only need 6 because... let me recount.
    
    # Actually j ranges over 0..m-1 = 0..2 (3 values), k ranges over 0..p-1 = 0..2 (3 values)
    # So there are 3*3 = 9 (j,k) pairs, not 6. Let me redo.
    
    # For each (j, k) pair, create one term:
    # U[i, j, r] = 1 if j matches, else 0
    # V[j, k, r] = 1 if k matches, else 0  
    # W[k, i, r] = 1 if k matches, else 0
    
    # Then sum over r: for fixed (i,j,k), only r with j_r=j and k_r=k contributes
    # That's exactly one r, contributing 1*1*1 = 1. Correct!
    
    rank = m * p  # 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for k in range(p):
            # U[i, j, r] = 1 for all i
            for i in range(n):
                U[i * m + j][r] = 1
            # V[j, k, r] = 1
            V[j * p + k][r] = 1
            # W[k, i, r] = 1 for all i
            for i in range(n):
                W[k * n + i][r] = 1
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

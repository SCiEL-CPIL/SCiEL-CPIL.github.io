"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Rank-5 decomposition of the all-ones 2x2x5 tensor
    # T[a,b,c] = 1 for all a in {0,1}, b in {0,1}, c in {0,1,2,3,4}
    # 
    # For each c = 0..4, use one term r = c:
    #   U[a*2+b, c] = 1 for all a, b
    #   V[b*5+c, c] = 1 for all b (and 0 for other c')
    #   W[c*2+a, c] = 1 for all a (and 0 for other c')
    #
    # Then sum_r U[a*2+b,r] * V[b*5+c,r] * W[c*2+a,r] = 1*1*1 = 1 (only r=c contributes)
    
    rank = 5
    U = [[0] * rank for _ in range(n * m)]  # 4 x 5
    V = [[0] * rank for _ in range(m * p)]  # 10 x 5
    W = [[0] * rank for _ in range(p * n)]  # 10 x 5
    
    for c in range(p):
        r = c
        # U: all ones in this column
        for a in range(n):
            for b in range(m):
                U[a * m + b][r] = 1
        # V: 1 at positions (b, c) for all b
        for b in range(m):
            V[b * p + c][r] = 1
        # W: 1 at positions (c, a) for all a
        for a in range(n):
            W[c * n + a][r] = 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for a in range(n):
        for b in range(m):
            for c in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[a * m + b, r] * V_arr[b * p + c, r] * W_arr[c * n + a, r]
                assert abs(val - 1.0) < 1e-9, f"Failed at ({a},{b},{c}): got {val}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

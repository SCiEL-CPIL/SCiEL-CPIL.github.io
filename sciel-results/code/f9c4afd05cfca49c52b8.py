"""Program for matrix multiplication 2x4 by 4x4 using Kronecker product structure."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 x 2x2
    strassen_U = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    # Rank-2 decomposition for 1x2 x 2x2
    # C[i, k] = sum_j A[i, j] * B[j, k] where A is 1x2, B is 2x2, C is 1x2
    # We use a rank-2 scheme:
    # M1 = A[0,0]*B[0,0] + A[0,1]*B[1,0]  -> contributes to C[0,0]
    # M2 = A[0,0]*B[0,1] + A[0,1]*B[1,1]  -> contributes to C[0,1]
    # This is just the naive 2-term decomposition for this small case.
    
    rank2_U = [
        {(0, 0): 1},  # term 0: picks A[0,0]
        {(0, 1): 1}   # term 1: picks A[0,1]
    ]
    rank2_V = [
        {(0, 0): 1, (1, 0): 1},  # term 0: B[0,0] and B[1,0]
        {(0, 1): 1, (1, 1): 1}   # term 1: B[0,1] and B[1,1]
    ]
    rank2_W = [
        {(0, 0): 1},  # term 0: C[0,0]
        {(1, 0): 1}   # term 1: C[0,1]
    ]
    
    # Build the Kronecker product decomposition
    # For each Strassen term s (0..6) and each rank-2 term t (0..1):
    #   r = s * 2 + t
    
    for s in range(7):
        for t in range(2):
            r = s * 2 + t
            
            # U: index is i * m + j where m=4, so i*4 + j
            # j = j1 * 2 + j2 where j1 is Strassen's j, j2 is rank-2's j
            # i is from Strassen
            for (i, j1), val_s in strassen_U[s].items():
                for (i2, j2), val_r2 in rank2_U[t].items():
                    # i2 should be 0 for rank-2 (1x2 matrix)
                    # The global i is i (from Strassen, 0 or 1)
                    # The global j is j1 * 2 + j2
                    j = j1 * 2 + j2
                    u_idx = i * m + j
                    U[u_idx][r] += val_s * val_r2
            
            # V: index is j * p + k where p=4, so j*4 + k
            # j = j1 * 2 + j2, k = k1 * 2 + k2
            for (j1, k1), val_s in strassen_V[s].items():
                for (j2, k2), val_r2 in rank2_V[t].items():
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v_idx = j * p + k
                    V[v_idx][r] += val_s * val_r2
            
            # W: index is k * n + i where n=2, so k*2 + i
            # k = k1 * 2 + k2, i is from Strassen
            for (i, k1), val_s in strassen_W[s].items():
                for (k2, i2), val_r2 in rank2_W[t].items():
                    # i2 should be 0 for rank-2 (1x2 output)
                    k = k1 * 2 + k2
                    w_idx = k * n + i
                    W[w_idx][r] += val_s * val_r2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

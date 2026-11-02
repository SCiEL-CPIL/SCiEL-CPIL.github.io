"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen local coefficients for 2x2 multiplication
    # U_loc[i, j], V_loc[j, k], W_loc[i, k]
    # 7 terms
    strassen_U = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): 1},  # M2
        {(0,0): 1},            # M3
        {(1,1): 1},            # M4
        {(0,0): 1, (0,1): 1},  # M5
        {(1,0): 1, (0,0): -1}, # M6
        {(0,1): 1, (1,1): -1}  # M7
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},  # M1
        {(0,0): 1},            # M2
        {(0,1): 1, (1,1): -1}, # M3
        {(1,0): 1, (0,0): -1}, # M4
        {(1,1): 1},            # M5
        {(0,0): 1, (0,1): 1},  # M6
        {(1,0): 1, (1,1): 1}   # M7
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): -1}, # M2
        {(0,1): 1, (1,1): 1},  # M3
        {(0,0): 1, (1,0): 1},  # M4
        {(0,0): -1, (0,1): 1}, # M5
        {(1,1): 1},            # M6
        {(0,0): 1}             # M7
    ]
    
    # 4 blocks
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 1: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    for b in range(4):
        for s in range(7):
            r = b * 7 + s
            
            # Map local (il, jl) to global (i, j)
            # Map local (jl, kl) to global (j, k)
            # Map local (il, kl) to global (i, k)
            
            # Determine offsets
            # b=0: j_off=0, k_off=0
            # b=1: j_off=2, k_off=0
            # b=2: j_off=0, k_off=2
            # b=3: j_off=2, k_off=2
            
            j_off = (b % 2) * 2
            k_off = (b // 2) * 2
            
            # Fill U
            for (il, jl), val in strassen_U[s].items():
                i = il
                j = jl + j_off
                U[i * m + j][r] += val
                
            # Fill V
            for (jl, kl), val in strassen_V[s].items():
                j = jl + j_off
                k = kl + k_off
                V[j * p + k][r] += val
                
            # Fill W
            for (il, kl), val in strassen_W[s].items():
                i = il
                k = kl + k_off
                W[k * n + i][r] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

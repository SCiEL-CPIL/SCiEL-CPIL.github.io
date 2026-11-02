"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U shape (n*m, rank) = (8, 28)
    # V shape (m*p, rank) = (16, 28)
    # W shape (p*n, rank) = (8, 28)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # u_s, v_s, w_s are length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1]
    
    u3 = [1, 0, 0, 0]
    v3 = [0, 0, 1, -1]
    w3 = [0, 1, 0, 1]
    
    u4 = [0, 0, 0, 1]
    v4 = [-1, 1, 0, 0]
    w4 = [-1, 1, 0, 0]
    
    u5 = [1, 0, 1, 0]
    v5 = [0, 0, 0, 1]
    w5 = [1, 0, 1, 0]
    
    u6 = [-1, 0, 0, 1]
    v6 = [1, 0, 1, 0]
    w6 = [0, 0, 0, 1]
    
    u7 = [0, 1, 0, -1]
    v7 = [0, 1, 0, 1]
    w7 = [1, 0, 0, 1]
    
    strassen_terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Blocks:
    # Block 0: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 1: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 3: i in {0,1}, j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0, 0), # i_offset, j_offset, k_offset
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2)
    ]
    
    for b_idx, (i_off, j_off, k_off) in enumerate(blocks):
        for t_idx, (u, v, w) in enumerate(strassen_terms):
            r = b_idx * 7 + t_idx
            
            # Map local u indices (0..3) to global U indices
            # Local (li, lj) -> li*2 + lj
            # Global (gi, gj) -> gi*4 + gj
            # gi = li + i_off, gj = lj + j_off
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    val = u[local_idx]
                    if val != 0:
                        gi = li + i_off
                        gj = lj + j_off
                        g_idx = gi * m + gj # gi*4 + gj
                        U[g_idx][r] += val
                        
            # Map local v indices (0..3) to global V indices
            # Local (lj, lk) -> lj*2 + lk
            # Global (gj, gk) -> gj*4 + gk
            # gj = lj + j_off, gk = lk + k_off
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    val = v[local_idx]
                    if val != 0:
                        gj = lj + j_off
                        gk = lk + k_off
                        g_idx = gj * p + gk # gj*4 + gk
                        V[g_idx][r] += val
                        
            # Map local w indices (0..3) to global W indices
            # Local (lk, li) -> lk*2 + li
            # Global (gk, gi) -> gk*2 + gi
            # gk = lk + k_off, gi = li + i_off
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    val = w[local_idx]
                    if val != 0:
                        gk = lk + k_off
                        gi = li + i_off
                        g_idx = gk * n + gi # gk*2 + gi
                        W[g_idx][r] += val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

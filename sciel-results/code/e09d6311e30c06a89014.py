"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    # Rank 49 decomposition using Kronecker product of 7 (Strassen) and 7 (blocks)
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Blocks of B: 14 / 2 = 7 blocks of size 2x2
    num_blocks = p // m  # 7
    strassen_rank = 7
    rank = num_blocks * strassen_rank # 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 * 2x2 -> 2x2
    # Indices: 00, 01, 10, 11
    U_S = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    V_S = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    W_S = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, 0], # M2 -> C10
        [0, 1, 0, -1], # M3 -> C01, C11
        [-1, 0, 1, 0], # M4 -> C00, C10
        [0, 1, 0, 0], # M5 -> C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # Fill U, V, W
    # U depends only on A (2x2), so same for all blocks
    # U[i*m+j, col]
    for s in range(strassen_rank):
        for u_idx in range(n * m): # 0..3
            val = U_S[u_idx][s]
            if val != 0:
                for q in range(num_blocks):
                    col = q * strassen_rank + s
                    U[u_idx][col] = val
    
    # V depends on B blocks
    # V[j*p+k, col]
    # k = 2*q + k_local
    # v_local = j*2 + k_local
    for q in range(num_blocks):
        for s in range(strassen_rank):
            col = q * strassen_rank + s
            for j in range(m): # 0..1
                for k_local in range(m): # 0..1
                    v_local = j * m + k_local
                    val = V_S[v_local][s]
                    if val != 0:
                        k = q * m + k_local
                        v_idx = j * p + k
                        V[v_idx][col] = val
                        
    # W depends on C blocks
    # W[k*n+i, col]
    # k = 2*q + k_local
    # w_local = k_local*2 + i
    for q in range(num_blocks):
        for s in range(strassen_rank):
            col = q * strassen_rank + s
            for k_local in range(m): # 0..1
                for i in range(n): # 0..1
                    w_local = k_local * n + i
                    val = W_S[w_local][s]
                    if val != 0:
                        k = q * m + k_local
                        w_idx = k * n + i
                        W[w_idx][col] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen's algorithm for 4 independent 2x2 blocks.
    # The tensor M_{2,2,8} decomposes into 4 disjoint blocks of M_{2,2,2}.
    # Rank(M_{2,2,2}) = 7 (Strassen). Total Rank = 4 * 7 = 28.
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # U (A): 7 linear combinations
    U_defs = [
        [(0, 1), (3, 1)],  # A00 + A11
        [(2, 1), (3, 1)],  # A10 + A11
        [(0, 1)],          # A00
        [(3, 1)],          # A11
        [(0, 1), (1, 1)],  # A00 + A01
        [(2, 1), (0, -1)], # A10 - A00
        [(1, 1), (3, -1)]  # A01 - A11
    ]
    
    # V (B block): 7 linear combinations for each block
    V_defs = [
        [(0, 1), (3, 1)],  # B00 + B11
        [(0, 1)],          # B00
        [(1, 1), (3, -1)], # B01 - B11
        [(2, 1), (0, -1)], # B10 - B00
        [(3, 1)],          # B11
        [(0, 1), (1, 1)],  # B00 + B01
        [(2, 1), (3, 1)]   # B10 + B11
    ]
    
    # W (C block): 7 linear combinations
    W_defs = [
        [(0, 1), (3, 1)],  # C00 + C11
        [(1, 1), (3, -1)], # C10 - C11
        [(2, 1), (3, 1)],  # C01 + C11
        [(0, 1), (1, 1)],  # C00 + C10
        [(0, -1), (2, 1)], # -C00 + C01
        [(3, 1)],          # C11
        [(0, 1)]           # C00
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            # Fill U
            for idx, val in U_defs[s]:
                U[idx][r] = val
                
            # Fill V
            # Local B index mapping:
            # 0: j=0, k=2q
            # 1: j=0, k=2q+1
            # 2: j=1, k=2q
            # 3: j=1, k=2q+1
            for local_idx, val in V_defs[s]:
                j = local_idx // 2
                k_local = local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
                
            # Fill W
            # Local C index mapping (trace convention W[k*n+i, r]):
            # 0: k=2q, i=0
            # 1: k=2q, i=1
            # 2: k=2q+1, i=0
            # 3: k=2q+1, i=1
            for local_idx, val in W_defs[s]:
                k_local = local_idx // 2
                i = local_idx % 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

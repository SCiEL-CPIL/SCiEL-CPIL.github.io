"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    local_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {1: 1}),
        ({0: 1}, {2: 1, 3: -1}, {2: 1, 3: 1}),
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 1: 1}),
        ({0: 1, 1: 1}, {3: 1}, {2: 1, 0: -1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    for b in range(4):
        for t in range(7):
            col = 7 * b + t
            u_dict, v_dict, w_dict = local_terms[t]
            
            for u_idx, val in u_dict.items():
                U[u_idx][col] += val
                
            for v_idx_local, val in v_dict.items():
                j = v_idx_local // 2
                k_local = v_idx_local % 2
                k_global = 2 * b + k_local
                v_idx_global = j * 8 + k_global
                V[v_idx_global][col] += val
                
            for w_idx_local, val in w_dict.items():
                k_local = w_idx_local // 2
                i = w_idx_local % 2
                k_global = 2 * b + k_local
                w_idx_global = k_global * 2 + i
                W[w_idx_global][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

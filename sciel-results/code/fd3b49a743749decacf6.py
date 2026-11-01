"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen factors for 2x2x2 multiplication
    # u, v, w are flattened 2x2 matrices (row-major: [00, 01, 10, 11])
    U_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    V_strassen = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    W_strassen = [
        [1, 0, 0, 1],  # M1 -> c00, c11
        [0, 0, 1, -1], # M2 -> c10, -c11
        [0, 1, 0, 1],  # M3 -> c01, c11
        [1, 0, 1, 0],  # M4 -> c00, c10
        [-1, 1, 0, 0], # M5 -> -c00, c01
        [0, 0, 0, 1],  # M6 -> c11
        [1, 0, 0, 0]   # M7 -> c00
    ]
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # s: 0 or 1 (block of m=4)
    # t: 0, 1 or 2 (block of p=6)
    for s in range(2):
        for t in range(3):
            base_col = (s * 3 + t) * 7
            for r in range(7):
                col = base_col + r
                u_vec = U_strassen[r]
                v_vec = V_strassen[r]
                w_vec = W_strassen[r]
                
                # Map local 2x2 indices to global indices
                # Local i_loc in 0..1, k_loc in 0..1, j_loc in 0..1
                for i_loc in range(2):
                    for k_loc in range(2):
                        # Global A index: i * m + k
                        # k_global = 2*s + k_loc
                        i_global = i_loc
                        k_global = 2 * s + k_loc
                        u_idx = i_global * m + k_global
                        U[u_idx][col] += u_vec[i_loc * 2 + k_loc]
                        
                        # Global C index: j * n + i
                        # j_global = 2*t + j_loc (for W mapping, but W depends on j_loc and i_loc)
                        # W is filled in separate loop or here?
                        # Let's fill W in a separate loop for clarity or combined.
                        # Combined:
                        for j_loc in range(2):
                            j_global = 2 * t + j_loc
                            w_idx = j_global * n + i_global
                            W[w_idx][col] += w_vec[i_loc * 2 + j_loc]
                            
                for k_loc in range(2):
                    for j_loc in range(2):
                        # Global B index: k * p + j
                        k_global = 2 * s + k_loc
                        j_global = 2 * t + j_loc
                        v_idx = k_global * p + j_global
                        V[v_idx][col] += v_vec[k_loc * 2 + j_loc]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

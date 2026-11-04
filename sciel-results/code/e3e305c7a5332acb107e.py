"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2 matrix multiplication
    # Indices for local 2x2:
    # U: i*2+j
    # V: j*2+k
    # W: k*2+i
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 0], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Iterate over 4 blocks corresponding to (j', k') in {0,1}x{0,1}
    for jp in range(2):
        for kp in range(2):
            for r in range(7):
                col = (jp * 2 + kp) * 7 + r
                
                # Fill U
                for l in range(4):
                    val = strassen_u[r][l]
                    if val != 0:
                        i = l // 2
                        j_loc = l % 2
                        j_global = jp * 2 + j_loc
                        u_idx = i * m + j_global
                        U[u_idx][col] = val
                        
                # Fill V
                for l in range(4):
                    val = strassen_v[r][l]
                    if val != 0:
                        j_loc = l // 2
                        k_loc = l % 2
                        j_global = jp * 2 + j_loc
                        k_global = kp * 2 + k_loc
                        v_idx = j_global * p + k_global
                        V[v_idx][col] = val
                        
                # Fill W
                for l in range(4):
                    val = strassen_w[r][l]
                    if val != 0:
                        k_loc = l // 2
                        i = l % 2
                        k_global = kp * 2 + k_loc
                        w_idx = k_global * n + i
                        W[w_idx][col] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28  # 4 Strassen instances * 7
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen factors for 2x2x2
    # U_local[x_i, x_j, c] for c in 0..6
    U_local = [
        [1, 0, 1, 0, 1, -1, 0],  # x_i=0, x_j=0
        [0, 0, 0, 0, 1, 0, 1],   # x_i=0, x_j=1
        [0, 1, 0, 1, 0, 1, -1],  # x_i=1, x_j=0
        [1, 1, 0, 1, 0, 0, -1],  # x_i=1, x_j=1
    ]
    
    # V_local[y_j, y_k, c]
    V_local = [
        [1, 1, 0, -1, 0, 1, 0],  # y_j=0, y_k=0
        [0, 0, 1, 0, 0, 1, 0],   # y_j=0, y_k=1
        [1, 0, -1, 1, 1, 0, 1],  # y_j=1, y_k=0
        [1, 0, -1, 0, 1, 0, 1],  # y_j=1, y_k=1
    ]
    
    # W_local[z_i, z_k, c]
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],  # z_i=0, z_k=0
        [0, 0, 1, 0, 1, 0, 0],   # z_i=0, z_k=1
        [0, 1, 0, 1, 0, 0, 0],   # z_i=1, z_k=0
        [1, -1, 1, 0, 0, 1, 0],  # z_i=1, z_k=1
    ]
    
    # 4 instances: (j_offset, k_offset)
    for inst in range(4):
        j_off = (inst // 2) * 2  # 0 or 2
        k_off = (inst % 2) * 2   # 0 or 2
        c_base = inst * 7
        
        for xi in range(2):
            for xj in range(2):
                for c in range(7):
                    val = U_local[xi][xj][c]
                    if val != 0:
                        u_idx = xi * m + (j_off + xj)
                        U[u_idx][c_base + c] += val
        
        for yj in range(2):
            for yk in range(2):
                for c in range(7):
                    val = V_local[yj][yk][c]
                    if val != 0:
                        v_idx = (j_off + yj) * p + (k_off + yk)
                        V[v_idx][c_base + c] += val
        
        for zi in range(2):
            for zk in range(2):
                for c in range(7):
                    val = W_local[zi][zk][c]
                    if val != 0:
                        w_idx = (k_off + zk) * n + zi
                        W[w_idx][c_base + c] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import time

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for one 2x2x2 block
    # U_local[4, 7], V_local[4, 7], W_local[4, 7]
    U_local = [
        [1, 1, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, -1, 0],
        [0, 0, 1, 1, 1, 0, -1]
    ]
    V_local = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 1, 0]
    ]
    W_local = [
        [0, -1, 0, 1, 1, 1, 0],
        [-1, 0, 1, 0, 1, -1, 0],
        [1, 1, 0, 0, 0, 0, -1],
        [0, 0, -1, 1, 0, 0, 1]
    ]
    
    # Map to global indices
    for q in range(4):
        for r_local in range(7):
            r_global = 7 * q + r_local
            
            # U: index is same
            for i in range(4):
                U[i][r_global] = U_local[i][r_local]
                
            # V: local index l -> global index l + 2*q
            for l in range(4):
                g_idx = l + 2 * q
                V[g_idx][r_global] = V_local[l][r_local]
                
            # W: local index l (2*k' + i) -> global index 16*q + 8*k' + i
            for l in range(4):
                k_prime = l // 2
                i = l % 2
                g_idx = 16 * q + 8 * k_prime + i
                W[g_idx][r_global] = W_local[l][r_local]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

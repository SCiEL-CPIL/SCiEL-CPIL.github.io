"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for first two columns (cols 0, 1)
    # U_S rows: a, b, c, d (indices 0, 1, 2, 3)
    # V_S rows: x, y, u, v (indices 0, 1, 3, 4)
    # W_S rows: C00, C10, C01, C11 (indices 0, 1, 2, 3)
    
    U_S = [
        [1, 0, 0, 0, 1, 0, 0], # a
        [1, 0, 1, 0, 0, 1, 0], # b
        [0, 1, 0, 0, 1, 0, 1], # c
        [0, 1, 0, 1, 0, 0, 1]  # d
    ]
    V_S = [
        [1, 1, 0, 0, 1, 0, 0], # x
        [0, 0, 1, 0, 0, 1, 0], # y
        [1, 1, -1, -1, 1, 0, 0], # u
        [0, 0, 0, 1, 0, 1, 0]  # v
    ]
    W_S = [
        [1, 0, 0, -1, 0, 0, 1], # C00
        [0, 1, 0, 0, 0, 0, 1],  # C10
        [0, 0, 1, 0, 0, 1, 0],  # C01
        [0, 0, 0, 1, 0, 0, 0]   # C11
    ]
    
    # Map Strassen to U, V, W
    # U indices: 0:a, 1:b, 2:c, 3:d
    for i in range(4):
        for r in range(7):
            U[i][r] = U_S[i][r]
            
    # V indices: 0:x, 1:y, 2:z, 3:u, 4:v, 5:w
    # Strassen V_S rows: x(0), y(1), u(2), v(3)
    # Map to V indices: x->0, y->1, u->3, v->4
    v_map = [0, 1, 3, 4]
    for i in range(4):
        for r in range(7):
            V[v_map[i]][r] = V_S[i][r]
            
    # W indices: 0:C00, 1:C10, 2:C01, 3:C11, 4:C02, 5:C12
    # Strassen W_S rows: C00(0), C10(1), C01(2), C11(3)
    w_map = [0, 1, 2, 3]
    for i in range(4):
        for r in range(7):
            W[w_map[i]][r] = W_S[i][r]
            
    # Schoolbook part for third column (col 2)
    # Terms 7, 8, 9, 10
    # a*z -> C02
    U[0][7] = 1
    V[2][7] = 1 # z
    W[4][7] = 1 # C02
    
    # b*w -> C02
    U[1][8] = 1
    V[5][8] = 1 # w
    W[4][8] = 1 # C02
    
    # c*z -> C12
    U[2][9] = 1
    V[2][9] = 1 # z
    W[5][9] = 1 # C12
    
    # d*w -> C12
    U[3][10] = 1
    V[5][10] = 1 # w
    W[5][10] = 1 # C12
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

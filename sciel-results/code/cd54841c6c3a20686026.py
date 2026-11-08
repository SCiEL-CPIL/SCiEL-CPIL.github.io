"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # a_idx: list of 4 indices for A (a,b,c,d)
    # b_idx: list of 4 indices for B (e,f,g,h)
    # w_offset: list of 4 indices for W (C00, C01, C10, C11)
    def add_strassen(a_idx, b_idx, w_idx):
        nonlocal col
        # m1: (a+d)(e+h)
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[0]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # m2: (c+d)e
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # m3: a(f-h)
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[w_idx[1]][col] += 1
        W[w_idx[3]][col] -= 1
        col += 1
        
        # m4: d(g-e)
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[w_idx[0]][col] -= 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # m5: (a+b)h
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # m6: (c-a)(e+f)
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # m7: (b-d)(g+h)
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1

    # Helper to add a 2x2 by 2x1 block
    # a_idx: list of 4 indices for A (a,b,c,d)
    # b_idx: list of 2 indices for B (e,f)
    # w_idx: list of 2 indices for W (C0, C1)
    def add_mat_vec(a_idx, b_idx, w_idx):
        nonlocal col
        # ae
        U[a_idx[0]][col] += 1
        V[b_idx[0]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        # bf
        U[a_idx[1]][col] += 1
        V[b_idx[1]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        # ce
        U[a_idx[2]][col] += 1
        V[b_idx[0]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        # df
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1

    # Define indices
    # A indices (row-major 2x4)
    # A1: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    a1_idx = [0, 1, 4, 5]
    # A2: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    a2_idx = [2, 3, 6, 7]
    
    # B indices (row-major 4x5)
    # B11: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 5, 6
    b11_idx = [0, 1, 5, 6]
    # B12: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 7, 8
    b12_idx = [2, 3, 7, 8]
    # B13: (0,4), (1,4) -> 4, 9
    b13_idx = [4, 9]
    
    # B21: (2,0), (2,1), (3,0), (3,1) -> 10, 11, 15, 16
    b21_idx = [10, 11, 15, 16]
    # B22: (2,2), (2,3), (3,2), (3,3) -> 12, 13, 17, 18
    b22_idx = [12, 13, 17, 18]
    # B23: (2,4), (3,4) -> 14, 19
    b23_idx = [14, 19]
    
    # W indices (col-major 2x5) W[k*n + i]
    # C11 (from A1B11): (0,0)->0, (0,1)->2, (1,0)->1, (1,1)->3
    w11_idx = [0, 2, 1, 3]
    # C12 (from A1B12): (0,2)->4, (0,3)->6, (1,2)->5, (1,3)->7
    w12_idx = [4, 6, 5, 7]
    # C13 (from A1B13): (0,4)->8, (1,4)->9
    w13_idx = [8, 9]
    
    # C21 (from A2B21): (0,0)->0, (0,1)->2, (1,0)->1, (1,1)->3
    w21_idx = [0, 2, 1, 3]
    # C22 (from A2B22): (0,2)->4, (0,3)->6, (1,2)->5, (1,3)->7
    w22_idx = [4, 6, 5, 7]
    # C23 (from A2B23): (0,4)->8, (1,4)->9
    w23_idx = [8, 9]

    # Add blocks
    add_strassen(a1_idx, b11_idx, w11_idx)
    add_strassen(a1_idx, b12_idx, w12_idx)
    add_mat_vec(a1_idx, b13_idx, w13_idx)
    
    add_strassen(a2_idx, b21_idx, w21_idx)
    add_strassen(a2_idx, b22_idx, w22_idx)
    add_mat_vec(a2_idx, b23_idx, w23_idx)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank 54:
    # Part 1: j in {0,1} -> 2x2x6 using 3x Strassen (Rank 21)
    # Part 2: j in {2,3,4} -> 2x3x6 split into 2x2x6 (Rank 21) + 2x1x6 (Rank 12)
    
    R = 54
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen Decomposition for 2x2x2
    # U: (i,j), V: (j,k), W: (k,i)
    # Indices: 00,01,10,11 -> 0,1,2,3
    
    Strassen_U = [
        [1, 0, 1, 0], # M1: A00+A11
        [0, 1, 0, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A10
        [-1, 0, 0, 1],# M6: A11-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    Strassen_V = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    Strassen_W = [
        [1, 0, 0, 1], # C00: M1, C11: M1
        [0, 0, 1, -1],# C10: M2, C11: -M2
        [0, 1, 0, 1], # C01: M3, C11: M3
        [-1, 0, 1, 0],# C00: -M4, C10: M4
        [1, 1, 0, 0], # C00: M5, C01: M5
        [0, 0, 0, 1], # C11: M6
        [-1, 0, 0, 0]  # C00: -M7
    ]
    # Note: W indices are k*n + i. n=2.
    # k=0,i=0 -> 0
    # k=1,i=0 -> 2
    # k=0,i=1 -> 1
    # k=1,i=1 -> 3
    # My W vectors above are indexed by [k,i]?
    # Let's re-verify Strassen_W mapping.
    # C00 (i=0,k=0, idx 0): M1(+), M4(-), M5(+), M7(-) -> 1, 0, 0, 1? No.
    # Col s corresponds to M_s.
    # W[k*n + i, s].
    
    # Let's construct W correctly per term s.
    # s=0 (M1): C00(+1), C11(+1). W[0]=1, W[3]=1.
    # s=1 (M2): C10(+1), C11(-1). W[2]=1, W[3]=-1.
    # s=2 (M3): C01(+1), C11(+1). W[1]=1, W[3]=1.
    # s=3 (M4): C00(-1), C10(+1). W[0]=-1, W[2]=1.
    # s=4 (M5): C00(+1), C01(+1). W[0]=1, W[1]=1.
    # s=5 (M6): C11(+1). W[3]=1.
    # s=6 (M7): C00(-1). W[0]=-1.
    
    Strassen_W = [
        [1, 0, 0, 1], # s=0
        [0, 0, 1, -1],# s=1
        [0, 1, 0, 1], # s=2
        [-1, 0, 1, 0],# s=3
        [1, 1, 0, 0], # s=4
        [0, 0, 0, 1], # s=5
        [-1, 0, 0, 0]  # s=6
    ]

    def add_strassen_block(u_offset, v_offset, w_offset, c_start):
        # Adds a 2x2x6 Strassen block
        # u_offset: global row index offset for U (i*5 + j_base) handled by loop
        # This function assumes we are adding to a specific slice of C
        # u_rows: list of (global_row, local_u_row)
        # v_rows: list of (global_row, local_v_row)
        # w_rows: list of (global_row, local_w_row)
        pass

    # Part 1: Global j in {0, 1}
    # Cols 0 to 20
    for b in range(3): # 3 blocks of 2 in k
        for s in range(7): # 7 Strassen terms
            c = b * 7 + s
            
            # U part
            for i in range(2):
                for j in range(2): # j_local 0,1 -> global j 0,1
                    u_local_idx = i * 2 + j
                    val = Strassen_U[s][u_local_idx]
                    if val != 0:
                        U[i * 5 + j][c] += val

            # V part
            for j in range(2): # j_local 0,1 -> global j 0,1
                for k_loc in range(2):
                    k = 2 * b + k_loc
                    v_local_idx = j * 2 + k_loc
                    val = Strassen_V[s][v_local_idx]
                    if val != 0:
                        V[j * 6 + k][c] += val

            # W part
            for k_loc in range(2):
                k = 2 * b + k_loc
                for i in range(2):
                    w_local_idx = k_loc * 2 + i # k*n + i
                    val = Strassen_W[s][w_local_idx]
                    if val != 0:
                        W[k * 2 + i][c] += val

    # Part 2: Global j in {2, 3, 4}
    # Subpart 2A: j in {2, 3} (local 0,1 of the 3-block)
    # Cols 21 to 41
    for b in range(3):
        for s in range(7):
            c = 21 + b * 7 + s
            
            # U part
            for i in range(2):
                for j_loc in range(2):
                    j = j_loc + 2 # global j
                    u_local_idx = i * 2 + j_loc
                    val = Strassen_U[s][u_local_idx]
                    if val != 0:
                        U[i * 5 + j][c] += val

            # V part
            for j_loc in range(2):
                j = j_loc + 2 # global j
                for k_loc in range(2):
                    k = 2 * b + k_loc
                    v_local_idx = j_loc * 2 + k_loc
                    val = Strassen_V[s][v_local_idx]
                    if val != 0:
                        V[j * 6 + k][c] += val

            # W part
            for k_loc in range(2):
                k = 2 * b + k_loc
                for i in range(2):
                    w_local_idx = k_loc * 2 + i
                    val = Strassen_W[s][w_local_idx]
                    if val != 0:
                        W[k * 2 + i][c] += val

    # Subpart 2B: j=4 (local 2 of the 3-block)
    # 2x1x6 Outer Product
    # Cols 42 to 53
    for i in range(2):
        for k in range(6):
            c = 42 + i * 6 + k
            j = 4
            U[i * 5 + j][c] = 1
            V[j * 6 + k][c] = 1
            W[k * 2 + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

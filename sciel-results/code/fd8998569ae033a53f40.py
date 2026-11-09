DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen_block(offset_u, offset_v, offset_w, start_col):
        # Strassen 2x2x2x2
        # U local: 4x7, V local: 4x7, W local: 4x7
        # Indices: i,j in 0..1; j,k in 0..1; k,i in 0..1
        s = [
            # M1
            ((0,0), (0,0), 1), ((1,1), (0,0), 1), ((0,0), (1,1), 1), ((1,1), (0,0), 1),
            # M2
            ((1,0), (0,0), 1), ((1,1), (0,0), 1), ((0,0), (1,1), 1), ((1,1), (0,0), 1),
            # M3
            ((0,0), (0,1), 1), ((0,1), (0,1), 1), ((0,0), (1,1), -1), ((1,1), (0,0), 1),
            # M4
            ((1,1), (1,0), 1), ((1,0), (1,0), 1), ((0,0), (1,1), -1), ((1,1), (0,0), 1),
            # M5
            ((0,0), (1,1), 1), ((0,1), (1,1), 1), ((0,0), (0,0), 1), ((1,1), (0,0), 1),
            # M6
            ((1,0), (0,0), -1), ((0,0), (0,0), 1), ((0,0), (0,1), 1), ((1,1), (0,0), 1),
            # M7
            ((0,1), (1,0), -1), ((1,1), (1,0), 1), ((0,1), (1,1), 1), ((1,1), (0,0), 1),
        ]
        # The above is messy. Let's use the standard formulas.
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need U, V, W factors for each M.
        # M1: U = A11+A22, V = B11+B22, W = C11+C22? No.
        # The term is M1 * W1.
        # W1 contributes to C11, C22.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for each M, we have a vector in U, V, and a matrix in W?
        # No, W is a vector in C space.
        # W1 = [1, 0, 0, 1] (for C11, C12, C21, C22)
        # W2 = [0, 0, 1, -1]
        # W3 = [0, 1, 0, 1]
        # W4 = [1, 0, 1, 0]
        # W5 = [-1, 1, 0, 0]
        # W6 = [0, 0, 0, 1]
        # W7 = [1, 0, 0, 0]
        
        # Let's verify.
        # C11 = M1 + M4 - M5 + M7
        # W1[0]=1, W4[0]=1, W5[0]=-1, W7[0]=1. Sum=1. Correct.
        # C12 = M3 + M5
        # W3[1]=1, W5[1]=1. Sum=1. Correct.
        # C21 = M2 + M4
        # W2[2]=1, W4[2]=1. Sum=1. Correct.
        # C22 = M1 - M2 + M3 + M6
        # W1[3]=1, W2[3]=-1, W3[3]=1, W6[3]=1. Sum=1. Correct.
        
        # Now U and V vectors.
        # M1: U = [1, 0, 0, 1] (A11, A12, A21, A22)
        #     V = [1, 0, 1, 0] (B11, B12, B21, B22)
        # M2: U = [0, 0, 1, 1]
        #     V = [1, 0, 0, 0]
        # M3: U = [1, 0, 0, 0]
        #     V = [0, 1, 0, -1]
        # M4: U = [0, 0, 0, 1]
        #     V = [0, 0, 1, -1]
        # M5: U = [1, 1, 0, 0]
        #     V = [0, 0, 0, 1]
        # M6: U = [0, 0, 1, -1]
        #     V = [1, 1, 0, 0]
        # M7: U = [0, 1, 0, -1]
        #     V = [0, 0, 1, 1]
        
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        
        for r in range(7):
            c = start_col + r
            for li in range(2):
                for lj in range(2):
                    U[offset_u + li * 2 + lj][c] += u_vecs[r][li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    V[offset_v + lj * 2 + lk][c] += v_vecs[r][lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    W[offset_w + lk * 2 + li][c] += w_vecs[r][lk * 2 + li]
                    
    # Helper to add schoolbook block 2x1x1x2
    def add_schoolbook_block(offset_u, offset_v, offset_w, start_col):
        # U: 2x1, V: 1x2, W: 2x2
        # r=0: i=0, k=0
        U[offset_u + 0][start_col + 0] += 1
        V[offset_v + 0][start_col + 0] += 1
        W[offset_w + 0][start_col + 0] += 1
        # r=1: i=0, k=1
        U[offset_u + 0][start_col + 1] += 1
        V[offset_v + 1][start_col + 1] += 1
        W[offset_w + 2][start_col + 1] += 1
        # r=2: i=1, k=0
        U[offset_u + 1][start_col + 2] += 1
        V[offset_v + 0][start_col + 2] += 1
        W[offset_w + 1][start_col + 2] += 1
        # r=3: i=1, k=1
        U[offset_u + 1][start_col + 3] += 1
        V[offset_v + 1][start_col + 3] += 1
        W[offset_w + 3][start_col + 3] += 1

    # Block 1: A[0:2, 0:2] B[0:2, 0:2] -> C[0:2, 0:2]
    # U offset: 0 (i=0,1; j=0,1) -> rows 0,1,3,4? No.
    # U row index: i * m + j. m=3.
    # i=0, j=0 -> 0. i=0, j=1 -> 1. i=1, j=0 -> 3. i=1, j=1 -> 4.
    # My add_strassen_block assumes contiguous rows.
    # I need to map local (i,j) to global row.
    
    # Let's rewrite the helpers to take mapping functions.
    
    def add_strassen_block_generic(i_map, j_map, k_map, start_col):
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        for r in range(7):
            c = start_col + r
            for li in range(2):
                for lj in range(2):
                    u_row = i_map(li) * m + j_map(lj)
                    U[u_row][c] += u_vecs[r][li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    v_row = j_map(lj) * p + k_map(lk)
                    V[v_row][c] += v_vecs[r][lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    w_row = k_map(lk) * n + i_map(li)
                    W[w_row][c] += w_vecs[r][lk * 2 + li]

    def add_schoolbook_block_generic(i_map, j_map, k_map, start_col):
        # i in 0..1, j in 0..0, k in 0..1
        for r in range(4):
            c = start_col + r
            li = r // 2
            lk = r % 2
            lj = 0
            u_row = i_map(li) * m + j_map(lj)
            v_row = j_map(lj) * p + k_map(lk)
            w_row = k_map(lk) * n + i_map(li)
            U[u_row][c] += 1
            V[v_row][c] += 1
            W[w_row][c] += 1

    # Block 1
    add_strassen_block_generic(lambda i: i, lambda j: j, lambda k: k, 0)
    # Block 2
    add_strassen_block_generic(lambda i: i, lambda j: j, lambda k: k + 2, 7)
    # Block 3
    add_schoolbook_block_generic(lambda i: i, lambda j: 2, lambda k: k, 14)
    # Block 4
    add_schoolbook_block_generic(lambda i: i, lambda j: 2, lambda k: k + 2, 18)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

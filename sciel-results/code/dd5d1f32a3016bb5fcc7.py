import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        for idx, coeff in u_entries.items():
            u_col[idx] = coeff
        for idx, coeff in v_entries.items():
            v_col[idx] = coeff
        for idx, coeff in w_entries.items():
            w_col[idx] = coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def merge_add(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result
        
    def merge_sub(d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) - v
        return result
        
    def merge_w(d1, d2, s1=1, s2=1):
        result = {}
        for k, v in d1.items():
            result[k] = result.get(k, 0) + s1 * v
        for k, v in d2.items():
            result[k] = result.get(k, 0) + s2 * v
        return result

    def a_idx(i, j):
        return i * m + j
    
    def b_idx(j, k):
        return j * p + k
    
    def c_idx(i, k):
        return k * n + i

    def strassen_virtual(va, vb, vc):
        a00, a01 = va[0][0], va[0][1]
        a10, a11 = va[1][0], va[1][1]
        b00, b01 = vb[0][0], vb[0][1]
        b10, b11 = vb[1][0], vb[1][1]
        c00, c01 = vc[0][0], vc[0][1]
        c10, c11 = vc[1][0], vc[1][1]
        
        add_term(
            merge_add(a00, a11),
            merge_add(b00, b11),
            merge_w(c00, c11, 1, 1)
        )
        add_term(
            merge_add(a10, a11),
            b00,
            merge_w(c10, c11, 1, -1)
        )
        add_term(
            a00,
            merge_sub(b01, b11),
            merge_w(c01, c11, 1, 1)
        )
        add_term(
            a11,
            merge_sub(b10, b00),
            merge_w(c00, c10, 1, 1)
        )
        add_term(
            merge_add(a00, a01),
            b11,
            merge_w(c00, c01, -1, 1)
        )
        add_term(
            merge_sub(a10, a00),
            merge_add(b00, b01),
            c11
        )
        add_term(
            merge_sub(a01, a11),
            merge_add(b10, b11),
            c00
        )
    
    def make_a_block(r1, r2, c1, c2):
        return [
            [{a_idx(r1, c1): 1}, {a_idx(r1, c2): 1}],
            [{a_idx(r2, c1): 1}, {a_idx(r2, c2): 1}]
        ]
    
    def make_b_block(r1, r2, c1, c2):
        return [
            [{b_idx(r1, c1): 1}, {b_idx(r1, c2): 1}],
            [{b_idx(r2, c1): 1}, {b_idx(r2, c2): 1}]
        ]
    
    def make_c_block(r1, r2, c1, c2):
        return [
            [{c_idx(r1, c1): 1}, {c_idx(r1, c2): 1}],
            [{c_idx(r2, c1): 1}, {c_idx(r2, c2): 1}]
        ]
    
    def block_add(b1, b2):
        return [
            [merge_add(b1[0][0], b2[0][0]), merge_add(b1[0][1], b2[0][1])],
            [merge_add(b1[1][0], b2[1][0]), merge_add(b1[1][1], b2[1][1])]
        ]
    
    def block_sub(b1, b2):
        return [
            [merge_sub(b1[0][0], b2[0][0]), merge_sub(b1[0][1], b2[0][1])],
            [merge_sub(b1[1][0], b2[1][0]), merge_sub(b1[1][1], b2[1][1])]
        ]
    
    # A blocks
    A11 = make_a_block(0, 1, 0, 1)
    A12 = make_a_block(0, 1, 2, 3)
    A21 = make_a_block(2, 3, 0, 1)
    A22 = make_a_block(2, 3, 2, 3)
    
    # B blocks for first 4 columns (cols 0-3)
    B11 = make_b_block(0, 1, 0, 1)
    B12 = make_b_block(0, 1, 2, 3)
    B21 = make_b_block(2, 3, 0, 1)
    B22 = make_b_block(2, 3, 2, 3)
    
    # C blocks for first 4 columns
    C11 = make_c_block(0, 1, 0, 1)
    C12 = make_c_block(0, 1, 2, 3)
    C21 = make_c_block(2, 3, 0, 1)
    C22 = make_c_block(2, 3, 2, 3)
    
    # C mappings for Strassen at 4x4 level
    def c_map_m1():
        return block_add(C11, C22)
    def c_map_m2():
        return [
            [merge_w(C21[0][0], C22[0][0], 1, -1), merge_w(C21[0][1], C22[0][1], 1, -1)],
            [merge_w(C21[1][0], C22[1][0], 1, -1), merge_w(C21[1][1], C22[1][1], 1, -1)]
        ]
    def c_map_m3():
        return [
            [merge_w(C12[0][0], C22[0][0], 1, 1), merge_w(C12[0][1], C22[0][1], 1, 1)],
            [merge_w(C12[1][0], C22[1][0], 1, 1), merge_w(C12[1][1], C22[1][1], 1, 1)]
        ]
    def c_map_m4():
        return [
            [merge_w(C11[0][0], C21[0][0], 1, 1), merge_w(C11[0][1], C21[0][1], 1, 1)],
            [merge_w(C11[1][0], C21[1][0], 1, 1), merge_w(C11[1][1], C21[1][1], 1, 1)]
        ]
    def c_map_m5():
        return [
            [merge_w(C11[0][0], C12[0][0], -1, 1), merge_w(C11[0][1], C12[0][1], -1, 1)],
            [merge_w(C11[1][0], C12[1][0], -1, 1), merge_w(C11[1][1], C12[1][1], -1, 1)]
        ]
    def c_map_m6():
        return C22
    def c_map_m7():
        return C11
    
    # Part 1: 4x4 x 4x4 (cols 0-3 of B) - recursive Strassen: 7*7 = 49 terms
    strassen_virtual(block_add(A11, A22), block_add(B11, B22), c_map_m1())
    strassen_virtual(block_add(A21, A22), B11, c_map_m2())
    strassen_virtual(A11, block_sub(B12, B22), c_map_m3())
    strassen_virtual(A22, block_sub(B21, B11), c_map_m4())
    strassen_virtual(block_add(A11, A12), B22, c_map_m5())
    strassen_virtual(block_sub(A21, A11), block_add(B11, B12), c_map_m6())
    strassen_virtual(block_sub(A12, A22), block_add(B21, B22), c_map_m7())
    
    # Part 2: 4x4 x 4x3 (cols 4-6 of B)
    # B blocks for cols 4-6
    BL_top = [
        [{b_idx(0, 4): 1}, {b_idx(0, 5): 1}, {b_idx(0, 6): 1}],
        [{b_idx(1, 4): 1}, {b_idx(1, 5): 1}, {b_idx(1, 6): 1}]
    ]
    BL_bot = [
        [{b_idx(2, 4): 1}, {b_idx(2, 5): 1}, {b_idx(2, 6): 1}],
        [{b_idx(3, 4): 1}, {b_idx(3, 5): 1}, {b_idx(3, 6): 1}]
    ]
    
    # C blocks for cols 4-6
    CL_top = [
        [{c_idx(0, 4): 1}, {c_idx(0, 5): 1}, {c_idx(0, 6): 1}],
        [{c_idx(1, 4): 1}, {c_idx(1, 5): 1}, {c_idx(1, 6): 1}]
    ]
    CL_bot = [
        [{c_idx(2, 4): 1}, {c_idx(2, 5): 1}, {c_idx(2, 6): 1}],
        [{c_idx(3, 4): 1}, {c_idx(3, 5): 1}, {c_idx(3, 6): 1}]
    ]
    
    # Schoolbook for each 2x2 x 2x3 subproblem (8 terms each)
    # Subproblem 1: A11 * BL_top -> CL_top
    for i in range(2):
        for j in range(2):
            for k in range(3):
                add_term(
                    A11[i][j],
                    BL_top[j][k],
                    CL_top[i][k]
                )
    
    # Subproblem 2: A12 * BL_bot -> CL_top
    for i in range(2):
        for j in range(2):
            for k in range(3):
                add_term(
                    A12[i][j],
                    BL_bot[j][k],
                    CL_top[i][k]
                )
    
    # Subproblem 3: A21 * BL_top -> CL_bot
    for i in range(2):
        for j in range(2):
            for k in range(3):
                add_term(
                    A21[i][j],
                    BL_top[j][k],
                    CL_bot[i][k]
                )
    
    # Subproblem 4: A22 * BL_bot -> CL_bot
    for i in range(2):
        for j in range(2):
            for k in range(3):
                add_term(
                    A22[i][j],
                    BL_bot[j][k],
                    CL_bot[i][k]
                )
    
    # Convert to final format
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float) and val == int(val):
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U = clean_matrix(U)
    V = clean_matrix(V)
    W = clean_matrix(W)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")

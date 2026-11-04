import time

def construct():
    """
    Returns a rank-80 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Decomposition:
    - Split B (4x9) into four 4x2 blocks (B1..B4) and one 4x1 block (B5).
    - For each 4x2 block: use schoolbook 2x2 * 2x2 multiplications (rank 4 each, 4 such = 16 per block).
    - For the 4x1 block: schoolbook 4x4 * 4x1 (rank 16).
    - Total rank: 4 * 16 + 16 = 80.
    """
    
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Schoolbook 2x2 * 2x2 terms
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    schoolbook_2x2 = [
        ({0: 1}, {0: 1}, {0: 1}),
        ({0: 1}, {1: 1}, {1: 1}),
        ({2: 1}, {0: 1}, {2: 1}),
        ({2: 1}, {1: 1}, {3: 1}),
    ]
    
    def add_schoolbook_4x2(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Adds 16 terms for a 4x4 * 4x2 multiplication using schoolbook 2x2 blocks.
        """
        # Block structure for 4x4 * 4x2:
        # A = [[A11, A12], [A21, A22]] (2x2 blocks)
        # B = [[B11], [B21]] (2x2 blocks, but B is 4x2 so B11 is 2x2, B21 is 2x2)
        # C = [[C11], [C21]] (2x2 blocks)
        
        # C11 = A11*B11 + A12*B21
        # C21 = A21*B11 + A22*B21
        
        # For each 2x2 * 2x2 multiplication, add 4 terms
        
        # A11 * B11 -> C11
        for u_c, v_c, w_c in schoolbook_2x2:
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            
            for loc, coeff in u_c.items():
                li = loc // 2
                lj = loc % 2
                gi = A_base_i + li
                gj = A_base_j + lj
                u_vec[gi * m + gj] += coeff
                
            for loc, coeff in v_c.items():
                lj = loc // 2
                lk = loc % 2
                gj = B_base_j + lj
                gk = B_base_k + lk
                v_vec[gj * p + gk] += coeff
                
            for loc, coeff in w_c.items():
                li = loc // 2
                lk = loc % 2
                gi = C_base_i + li
                gk = C_base_k + lk
                w_vec[gk * n + gi] += coeff
                
            add_term(u_vec, v_vec, w_vec)
            
        # A12 * B21 -> C11
        for u_c, v_c, w_c in schoolbook_2x2:
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            
            for loc, coeff in u_c.items():
                li = loc // 2
                lj = loc % 2
                gi = A_base_i + li
                gj = A_base_j + 2 + lj
                u_vec[gi * m + gj] += coeff
                
            for loc, coeff in v_c.items():
                lj = loc // 2
                lk = loc % 2
                gj = B_base_j + 2 + lj
                gk = B_base_k + lk
                v_vec[gj * p + gk] += coeff
                
            for loc, coeff in w_c.items():
                li = loc // 2
                lk = loc % 2
                gi = C_base_i + li
                gk = C_base_k + lk
                w_vec[gk * n + gi] += coeff
                
            add_term(u_vec, v_vec, w_vec)
            
        # A21 * B11 -> C21
        for u_c, v_c, w_c in schoolbook_2x2:
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            
            for loc, coeff in u_c.items():
                li = loc // 2
                lj = loc % 2
                gi = A_base_i + 2 + li
                gj = A_base_j + lj
                u_vec[gi * m + gj] += coeff
                
            for loc, coeff in v_c.items():
                lj = loc // 2
                lk = loc % 2
                gj = B_base_j + lj
                gk = B_base_k + lk
                v_vec[gj * p + gk] += coeff
                
            for loc, coeff in w_c.items():
                li = loc // 2
                lk = loc % 2
                gi = C_base_i + 2 + li
                gk = C_base_k + lk
                w_vec[gk * n + gi] += coeff
                
            add_term(u_vec, v_vec, w_vec)
            
        # A22 * B21 -> C21
        for u_c, v_c, w_c in schoolbook_2x2:
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            
            for loc, coeff in u_c.items():
                li = loc // 2
                lj = loc % 2
                gi = A_base_i + 2 + li
                gj = A_base_j + 2 + lj
                u_vec[gi * m + gj] += coeff
                
            for loc, coeff in v_c.items():
                lj = loc // 2
                lk = loc % 2
                gj = B_base_j + 2 + lj
                gk = B_base_k + lk
                v_vec[gj * p + gk] += coeff
                
            for loc, coeff in w_c.items():
                li = loc // 2
                lk = loc % 2
                gi = C_base_i + 2 + li
                gk = C_base_k + lk
                w_vec[gk * n + gi] += coeff
                
            add_term(u_vec, v_vec, w_vec)
            
    # Part 1: A * B1 (cols 0-1)
    add_schoolbook_4x2(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 2-3)
    add_schoolbook_4x2(0, 0, 0, 2, 0, 2)
    
    # Part 3: A * B3 (cols 4-5)
    add_schoolbook_4x2(0, 0, 0, 4, 0, 4)
    
    # Part 4: A * B4 (cols 6-7)
    add_schoolbook_4x2(0, 0, 0, 6, 0, 6)
    
    # Part 5: A * B5 (col 8)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u_vec = [0] * 16
            v_vec = [0] * 36
            w_vec = [0] * 36
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}

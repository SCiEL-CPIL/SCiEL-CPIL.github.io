import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # n=4, m=4, p=9
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 2x2 terms
    # Indices for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # u: A indices, v: B indices, w: C indices
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 - m4 + m6
    # c22 = -m1 + m2 + m3 + m6
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # Each entry is (u_dict, v_dict, w_dict)
    # u_dict: {index: coeff} for A
    # v_dict: {index: coeff} for B
    # w_dict: {index: coeff} for C
    
    strassen_2x2 = [
        # m1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # m2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # m3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # m4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # m5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # m6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        """
        Adds 49 terms for a 4x4 * 4x4 multiplication.
        A is 4x4, B is 4x4, C is 4x4.
        The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
        """
        # Helper to add one inner Strassen term
        def add_inner(A_blocks, B_blocks, C_blocks):
            # A_blocks: list of (br, bc, coeff) where br, bc are block row/col (0 or 1)
            # This maps to the 2x2 indices inside the block.
            
            for u_c, v_c, w_c in strassen_2x2:
                u_vec = [0] * 16
                v_vec = [0] * 36
                w_vec = [0] * 36
                
                # Map inner 2x2 indices to global A indices
                # Inner index loc (0..3) -> (li, lj) where li=loc//2, lj=loc%2
                # Global index in A: (A_base_i + br*2 + li, A_base_j + bc*2 + lj)
                for loc, coeff in u_c.items():
                    li = loc // 2
                    lj = loc % 2
                    for br, bc, bc_coeff in A_blocks:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[gi * m + gj] += coeff * bc_coeff
                        
                # Map inner 2x2 indices to global B indices
                # Inner index loc -> (lj, lk)
                # Global index in B: (B_base_j + br*2 + lj, B_base_k + bc*2 + lk)
                for loc, coeff in v_c.items():
                    lj = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in B_blocks:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[gj * p + gk] += coeff * bc_coeff
                        
                # Map inner 2x2 indices to global C indices
                # Inner index loc -> (li, lk)
                # Global index in C: (C_base_i + br*2 + li, C_base_k + bc*2 + lk)
                # W index: k*n + i
                for loc, coeff in w_c.items():
                    li = loc // 2
                    lk = loc % 2
                    for br, bc, bc_coeff in C_blocks:
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[gk * n + gi] += coeff * bc_coeff
                        
                add_term(u_vec, v_vec, w_vec)
                
        # Strassen 7 steps for 4x4 * 4x4
        # Blocks:
        # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
        # B11 (0,0), B12 (0,1), B21 (1,0), B22 (1,1)
        # C11 (0,0), C12 (0,1), C21 (1,0), C22 (1,1)
        
        # M1 = (A11+A22)(B11+B22) -> C11, C22
        add_inner(
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        )
        
        # M2 = (A21+A22)B11 -> C21, C22
        add_inner(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(1,0,1), (1,1,-1)]
        )
        
        # M3 = A11(B12-B22) -> C12, C22
        add_inner(
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        )
        
        # M4 = A22(B21-B11) -> C11, C21
        add_inner(
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        )
        
        # M5 = (A11+A12)B22 -> C11, C12
        add_inner(
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        )
        
        # M6 = (A21-A11)(B11+B12) -> C22
        add_inner(
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        )
        
        # M7 = (A12-A22)(B21+B22) -> C11
        add_inner(
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        )

    # Part 1: A * B1 (cols 0-3)
    # A is 4x4, B1 is 4x4 (cols 0-3), C1 is 4x4 (cols 0-3)
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    # A is 4x4, B2 is 4x4 (cols 4-7), C2 is 4x4 (cols 4-7)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8)
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
    
    if time.time() > deadline:
        pass

    U = [[U_cols[col][row] for col in range(rank)] for row in range(16)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(36)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(36)]
    
    return {"U": U, "V": V, "W": W}

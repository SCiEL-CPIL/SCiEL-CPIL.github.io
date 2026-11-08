import numpy as np

def construct():
    """
    Returns a rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    The decomposition splits B into three blocks:
    1. B[:, 0:4] -> C[:, 0:4] using Strassen 4x4 (rank 49)
    2. B[:, 4:8] -> C[:, 4:8] using Strassen 4x4 (rank 49)
    3. B[:, 8:9] -> C[:, 8:9] using Schoolbook 4x4*4x1 (rank 16)
    
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen 4x4 terms
    # We define the 49 terms for 4x4 * 4x4 -> 4x4
    # This is derived from 7 Strassen 2x2 meta-multiplications, each expanded into 7 2x2 multiplications.
    # Total 49 terms.
    
    # Helper to add a single rank-1 term for a 2x2 block multiplication
    # u_dict: {(i,j): coeff} for A block
    # v_dict: {(j,k): coeff} for B block
    # w_dict: {(i,k): coeff} for C block
    
    strassen_2x2 = [
        # (A, B, C) coefficients
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Block decomposition for 4x4
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # 7 meta-terms for 2x2 block matrix multiplication
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-A00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    meta_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_row, block_col, coeff)
        # B_blocks: list of (block_row, block_col, coeff)
        # C_blocks: list of (block_row, block_col, coeff)
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], { (0,0):1, (1,1):1 }),
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], { (1,0):1, (1,1):-1 }),
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], { (0,1):1, (1,1):1 }),
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], { (0,0):1, (1,0):1 }),
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], { (0,0):-1, (0,1):1 }),
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], { (1,1):1 }),
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], { (0,0):1 }),
    ]
    
    def add_strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        for A_blocks, B_blocks, C_blocks in meta_terms:
            for A_meta, B_meta, C_meta in strassen_2x2:
                u_vec = [0] * (n*m)
                v_vec = [0] * (m*p)
                w_vec = [0] * (p*n)
                
                # Expand A
                for (li, lj), coeff_a in A_meta.items():
                    for (br, bc, coeff_block) in A_blocks:
                        gi = A_base_i + br*2 + li
                        gj = A_base_j + bc*2 + lj
                        u_vec[gi*m + gj] += coeff_a * coeff_block
                        
                # Expand B
                for (lj, lk), coeff_b in B_meta.items():
                    for (br, bc, coeff_block) in B_blocks:
                        gj = B_base_j + br*2 + lj
                        gk = B_base_k + bc*2 + lk
                        v_vec[gj*p + gk] += coeff_b * coeff_block
                        
                # Expand C
                for (li, lk), coeff_c in C_meta.items():
                    for (br, bc, coeff_block) in C_blocks.items():
                        gi = C_base_i + br*2 + li
                        gk = C_base_k + bc*2 + lk
                        w_vec[gk*n + gi] += coeff_c * coeff_block
                        
                add_term(u_vec, v_vec, w_vec)

    # Block 1: B[:, 0:4]
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    
    # Block 2: B[:, 4:8]
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Block 3: B[:, 8:9] (Schoolbook)
    for i in range(n):
        for j in range(m):
            u_vec = [0] * (n*m)
            v_vec = [0] * (m*p)
            w_vec = [0] * (p*n)
            u_vec[i*m + j] = 1
            v_vec[j*p + 8] = 1
            w_vec[8*n + i] = 1
            add_term(u_vec, v_vec, w_vec)
            
    rank = len(U_cols)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for col in range(rank):
        for row in range(n*m):
            U[row, col] = U_cols[col][row]
        for row in range(m*p):
            V[row, col] = V_cols[col][row]
        for row in range(p*n):
            W[row, col] = W_cols[col][row]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

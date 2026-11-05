import numpy as np

def construct():
    n, m, p = 3, 4, 5
    rank = 56
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Helper to add Strassen block for 2x2x2
    # a_rows: indices of i in A (e.g., [0, 1])
    # a_cols: indices of j in A (e.g., [0, 1])
    # b_rows: indices of j in B (e.g., [0, 1])
    # b_cols: indices of k in B (e.g., [0, 1])
    # w_offset: offset for k in W index calculation? 
    # Actually, W index is k*n + i.
    # We need to specify which i and k are involved.
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, w_k_base):
        nonlocal col
        # a_rows: [i0, i1], a_cols: [j0, j1]
        # b_rows: [j0, j1], b_cols: [k0, k1]
        
        i0, i1 = a_rows
        j0, j1 = a_cols
        # Note: b_rows should be same as a_cols for matrix mult consistency
        k0, k1 = b_cols
        
        # 1. (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        U[i0*m + j0, col] += 1
        U[i1*m + j1, col] += 1
        V[j0*p + k0, col] += 1
        V[j1*p + k1, col] += 1
        # W: C[i0,k0] += 1, C[i0,k1] += 1, C[i1,k0] += 1, C[i1,k1] += 1
        W[k0*n + i0, col] += 1
        W[k1*n + i0, col] += 1
        W[k0*n + i1, col] += 1
        W[k1*n + i1, col] += 1
        col += 1
        
        # 2. (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[i1*m + j0, col] += 1
        U[i1*m + j1, col] += 1
        V[j0*p + k0, col] += 1
        # W: C[i1,k0] += 1
        W[k0*n + i1, col] += 1
        col += 1
        
        # 3. A[i0,j0] * (B[j0,k1] - B[j1,k1])
        U[i0*m + j0, col] += 1
        V[j0*p + k1, col] += 1
        V[j1*p + k1, col] -= 1
        # W: C[i0,k1] += 1
        W[k1*n + i0, col] += 1
        col += 1
        
        # 4. A[i1,j1] * (B[j1,k0] - B[j0,k0])
        U[i1*m + j1, col] += 1
        V[j1*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        # W: C[i1,k0] += 1
        W[k0*n + i1, col] += 1
        col += 1
        
        # 5. (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[i0*m + j0, col] += 1
        U[i0*m + j1, col] += 1
        V[j1*p + k1, col] += 1
        # W: C[i0,k1] -= 1 (from C00 formula? No, check formula)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M5 contributes to C01 (+1) and C00 (-1)
        W[k1*n + i0, col] += 1
        W[k0*n + i0, col] -= 1
        col += 1
        
        # 6. (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        U[i1*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + k1, col] += 1
        # W: C[i1,k1] += 1 (from C11 formula)
        W[k1*n + i1, col] += 1
        col += 1
        
        # 7. (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        U[i0*m + j1, col] += 1
        U[i1*m + j1, col] -= 1
        V[j1*p + k0, col] += 1
        V[j1*p + k1, col] += 1
        # W: C[i0,k0] += 1 (from C00 formula)
        W[k0*n + i0, col] += 1
        col += 1

    def add_schoolbook_3x2x2(a_row, a_cols, b_rows, b_cols, w_k_base):
        nonlocal col
        i = a_row
        j0, j1 = a_cols
        k0, k1 = b_cols
        
        # C[i,k0] = A[i,j0]B[j0,k0] + A[i,j1]B[j1,k0]
        # C[i,k1] = A[i,j0]B[j0,k1] + A[i,j1]B[j1,k1]
        
        # 8. A[i,j0] * B[j0,k0]
        U[i*m + j0, col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + i, col] += 1
        col += 1
        
        # 9. A[i,j1] * B[j1,k0]
        U[i*m + j1, col] += 1
        V[j1*p + k0, col] += 1
        W[k0*n + i, col] += 1
        col += 1
        
        # 10. A[i,j0] * B[j0,k1]
        U[i*m + j0, col] += 1
        V[j0*p + k1, col] += 1
        W[k1*n + i, col] += 1
        col += 1
        
        # 11. A[i,j1] * B[j1,k1]
        U[i*m + j1, col] += 1
        V[j1*p + k1, col] += 1
        W[k1*n + i, col] += 1
        col += 1

    def add_schoolbook_3x2x1(a_cols, b_rows, b_col):
        nonlocal col
        j0, j1 = a_cols
        k = b_col
        
        # C[i,k] = A[i,j0]B[j0,k] + A[i,j1]B[j1,k] for i=0,1,2
        for i in range(3):
            # A[i,j0] * B[j0,k]
            U[i*m + j0, col] += 1
            V[j0*p + k, col] += 1
            W[k*n + i, col] += 1
            col += 1
            
            # A[i,j1] * B[j1,k]
            U[i*m + j1, col] += 1
            V[j1*p + k, col] += 1
            W[k*n + i, col] += 1
            col += 1

    # Block 1: j in {0, 1}
    # Sub-block 1.1: k in {0, 1}
    # Strassen for i in {0, 1}
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], 0)
    # Schoolbook for i = 2
    add_schoolbook_3x2x2(2, [0, 1], [0, 1], [0, 1], 0)
    
    # Sub-block 1.2: k in {2, 3}
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], 2)
    add_schoolbook_3x2x2(2, [0, 1], [0, 1], [2, 3], 2)
    
    # Sub-block 1.3: k = 4
    add_schoolbook_3x2x1([0, 1], [0, 1], 4)
    
    # Block 2: j in {2, 3}
    # Sub-block 2.1: k in {0, 1}
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], 0)
    add_schoolbook_3x2x2(2, [2, 3], [2, 3], [0, 1], 0)
    
    # Sub-block 2.2: k in {2, 3}
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], 2)
    add_schoolbook_3x2x2(2, [2, 3], [2, 3], [2, 3], 2)
    
    # Sub-block 2.3: k = 4
    add_schoolbook_3x2x1([2, 3], [2, 3], 4)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

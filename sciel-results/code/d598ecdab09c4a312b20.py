import numpy as np

def construct():
    n, m, p = 3, 4, 5
    # We will build U, V, W lists of lists
    U = []
    V = []
    W = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is row in U (n*m)
        # v_idx is row in V (m*p)
        # w_idx is row in W (p*n)
        col = len(U)
        U.append([0] * (col + 1)) # This is inefficient, better to build columns
        # Let's build columns directly
        pass

    # Better approach: build U, V, W as lists of columns
    # U is (n*m) x rank
    # V is (m*p) x rank
    # W is (p*n) x rank
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_row, v_row, w_row, val=1):
        U_cols.append([0] * (n * m))
        V_cols.append([0] * (m * p))
        W_cols.append([0] * (p * n))
        U_cols[-1][u_row] = val
        V_cols[-1][v_row] = val
        W_cols[-1][w_row] = val

    def add_strassen_2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # a_row_offset: offset in A rows (0..2)
        # b_col_offset: offset in B cols (0..4)
        # c_row_offset: offset in C rows (0..2)
        # c_col_offset: offset in C cols (0..4)
        # A is 2x2, B is 2x2, C is 2x2
        # Indices in A: (r, c) -> r in 0..1, c in 0..1
        # Indices in B: (r, c) -> r in 0..1, c in 0..1
        # Indices in C: (r, c) -> r in 0..1, c in 0..1
        
        # Strassen 7 multiplications
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # We need to map these to the global indices
        # A[i, j] global index: (a_row_offset + i) * m + (b_col_offset + j) ? No.
        # A is n x m. B is m x p.
        # The 2x2 block of A is at rows a_row_offset..a_row_offset+1, cols b_col_offset..b_col_offset+1?
        # Wait, the inner dimension is m.
        # A is 3x4. B is 4x5.
        # In the 3x2x5 block:
        # A_block is 3x2. B_block is 2x5.
        # In the 2x2x2 sub-block:
        # A_sub is 2x2. B_sub is 2x2.
        
        # Let's define the local indices
        # A_sub[i, j] corresponds to global A[a_row_offset + i, b_col_offset + j]
        # Wait, b_col_offset is the column offset in A, which is the row offset in B.
        # So A_sub[i, j] is A[a_row_offset + i, b_col_offset + j]
        # B_sub[i, j] is B[b_col_offset + i, c_col_offset + j]
        # C_sub[i, j] is C[c_row_offset + i, c_col_offset + j]
        
        # U index for A[a, b] is a * m + b
        # V index for B[b, c] is b * p + c
        # W index for C[c, a] is c * n + a
        
        def u_idx(a, b): return a * m + b
        def v_idx(b, c): return b * p + c
        def w_idx(c, a): return c * n + a
        
        # M1
        # (A11 + A22) * (B11 + B22)
        # This is not a single multiplication term in the tensor decomposition.
        # The tensor decomposition is a sum of rank-1 tensors.
        # Strassen expresses the product as 7 rank-1 terms? No.
        # Strassen uses 7 multiplications, but the "multiplications" are of linear combinations.
        # The tensor decomposition rank is the number of such multiplications.
        # So we add 7 terms.
        
        # Term 1: (A11 + A22) * (B11 + B22)
        # This contributes to C11, C12, C21, C22.
        # C11 += M1, C12 += M1, C21 += M1, C22 += M1
        # Wait, the Strassen formulas are:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So we need to add terms with coefficients +1 or -1 to the W matrix.
        
        # Let's define the linear combinations for A and B for each M.
        # M1: A: A11+A22, B: B11+B22
        # M2: A: A21+A22, B: B11
        # M3: A: A11, B: B12-B22
        # M4: A: A22, B: B21-B11
        # M5: A: A11+A12, B: B22
        # M6: A: A21-A11, B: B11+B12
        # M7: A: A12-A22, B: B21+B22
        
        # We will create 7 new columns in U, V, W.
        
        # M1
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        # A part
        U_col[u_idx(a_row_offset + 0, b_col_offset + 0)] += 1
        U_col[u_idx(a_row_offset + 1, b_col_offset + 1)] += 1
        # B part
        V_col[v_idx(b_col_offset + 0, c_col_offset + 0)] += 1
        V_col[v_idx(b_col_offset + 1, c_col_offset + 1)] += 1
        # C part
        W_col[w_idx(c_col_offset + 0, c_row_offset + 0)] += 1 # C11
        W_col[w_idx(c_col_offset + 1, c_row_offset + 0)] += 1 # C12
        W_col[w_idx(c_col_offset + 0, c_row_offset + 1)] += 1 # C21
        W_col[w_idx(c_col_offset + 1, c_row_offset + 1)] += 1 # C22
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M2
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        U_col[u_idx(a_row_offset + 1, b_col_offset + 0)] += 1
        U_col[u_idx(a_row_offset + 1, b_col_offset + 1)] += 1
        V_col[v_idx(b_col_offset + 0, c_col_offset + 0)] += 1
        W_col[w_idx(c_col_offset + 0, c_row_offset + 1)] += 1 # C21
        W_col[w_idx(c_col_offset + 1, c_row_offset + 1)] -= 1 # C22
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M3
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        U_col[u_idx(a_row_offset + 0, b_col_offset + 0)] += 1
        V_col[v_idx(b_col_offset + 0, c_col_offset + 1)] += 1
        V_col[v_idx(b_col_offset + 1, c_col_offset + 1)] -= 1
        W_col[w_idx(c_col_offset + 1, c_row_offset + 0)] += 1 # C12
        W_col[w_idx(c_col_offset + 1, c_row_offset + 1)] += 1 # C22
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M4
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        U_col[u_idx(a_row_offset + 1, b_col_offset + 1)] += 1
        V_col[v_idx(b_col_offset + 1, c_col_offset + 0)] += 1
        V_col[v_idx(b_col_offset + 0, c_col_offset + 0)] -= 1
        W_col[w_idx(c_col_offset + 0, c_row_offset + 0)] += 1 # C11
        W_col[w_idx(c_col_offset + 0, c_row_offset + 1)] += 1 # C21
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M5
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        U_col[u_idx(a_row_offset + 0, b_col_offset + 0)] += 1
        U_col[u_idx(a_row_offset + 0, b_col_offset + 1)] += 1
        V_col[v_idx(b_col_offset + 1, c_col_offset + 1)] += 1
        W_col[w_idx(c_col_offset + 0, c_row_offset + 0)] -= 1 # C11
        W_col[w_idx(c_col_offset + 1, c_row_offset + 0)] += 1 # C12
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M6
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        U_col[u_idx(a_row_offset + 1, b_col_offset + 0)] += 1
        U_col[u_idx(a_row_offset + 0, b_col_offset + 0)] -= 1
        V_col[v_idx(b_col_offset + 0, c_col_offset + 0)] += 1
        V_col[v_idx(b_col_offset + 0, c_col_offset + 1)] += 1
        W_col[w_idx(c_col_offset + 1, c_row_offset + 1)] += 1 # C22
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
        
        # M7
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        U_col[u_idx(a_row_offset + 0, b_col_offset + 1)] += 1
        U_col[u_idx(a_row_offset + 1, b_col_offset + 1)] -= 1
        V_col[v_idx(b_col_offset + 1, c_col_offset + 0)] += 1
        V_col[v_idx(b_col_offset + 1, c_col_offset + 1)] += 1
        W_col[w_idx(c_col_offset + 0, c_row_offset + 0)] += 1 # C11
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def add_schoolbook_2x2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # A is 2x2, B is 2x2, C is 2x2
        # 8 terms
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    add_term(
                        u_idx(a_row_offset + i, b_col_offset + j),
                        v_idx(b_col_offset + j, c_col_offset + k),
                        w_idx(c_col_offset + k, c_row_offset + i)
                    )
                    
    # We need a generic schoolbook for arbitrary sizes
    def add_schoolblock(a_row_start, a_row_end, b_col_start, b_col_end, c_row_start, c_row_end):
        # A is a_row_end-a_row_start x b_col_end-b_col_start
        # B is b_col_end-b_col_start x c_row_end-c_row_start
        # C is c_row_end-c_row_start x a_row_end-a_row_start
        for i in range(a_row_start, a_row_end):
            for j in range(b_col_start, b_col_end):
                for k in range(c_row_start, c_row_end):
                    add_term(
                        i * m + j,
                        j * p + k,
                        k * n + i
                    )

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 0, 0, 0)
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A is 2x2, B is 2x2.
    # But B[2:4, 0:2] is rows 2,3 of B.
    # In our indexing, B is m x p.
    # b_col_offset is the column in A, which is the row in B.
    # So b_col_offset = 2.
    # c_col_offset = 0.
    # a_row_offset = 0.
    # c_row_offset = 0.
    add_schoolblock(0, 2, 2, 4, 0, 2) # 2x2x2 = 8 terms
    
    # Block 3: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    # A is 1x2, B is 2x2.
    add_schoolblock(2, 3, 0, 2, 2, 3) # 1x2x2 = 4 terms
    
    # Block 4: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    # A is 1x2, B is 2x2.
    add_schoolblock(2, 3, 2, 4, 2, 3) # 1x2x2 = 4 terms
    
    # Now for C[0:2, 2:5] and C[2:3, 2:5]
    # A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # A is 2x2, B is 2x3.
    add_schoolblock(0, 2, 0, 2, 0, 2) # 2x2x3 = 12 terms
    
    # A[0:2, 2:4] * B[2:4, 2:5] -> C[0:2, 2:5]
    # A is 2x2, B is 2x3.
    add_schoolblock(0, 2, 2, 4, 0, 2) # 2x2x3 = 12 terms
    
    # A[2:3, 0:2] * B[0:2, 2:5] -> C[2:3, 2:5]
    # A is 1x2, B is 2x3.
    add_schoolblock(2, 3, 0, 2, 2, 3) # 1x2x3 = 6 terms
    
    # A[2:3, 2:4] * B[2:4, 2:5] -> C[2:3, 2:5]
    # A is 1x2, B is 2x3.
    add_schoolblock(2, 3, 2, 4, 2, 3) # 1x2x3 = 6 terms

    # Total rank: 7 + 8 + 4 + 4 + 12 + 12 + 6 + 6 = 59?
    # 7 + 8 = 15
    # 15 + 4 = 19
    # 19 + 4 = 23
    # 23 + 12 = 35
    # 35 + 12 = 47
    # 47 + 6 = 53
    # 53 + 6 = 59.
    
    # Wait, I missed some blocks?
    # C is 3x5.
    # C[0:2, 0:2] covered by blocks 1,2,3,4?
    # Block 1: A[0:2, 0:2] B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[2:3, 0:2] B[0:2, 0:2] -> C[2:3, 0:2]
    # Block 4: A[2:3, 2:4] B[2:4, 0:2] -> C[2:3, 0:2]
    
    # C[0:2, 2:5]
    # Block 5: A[0:2, 0:2] B[0:2, 2:5] -> C[0:2, 2:5]
    # Block 6: A[0:2, 2:4] B[2:4, 2:5] -> C[0:2, 2:5]
    # Block 7: A[2:3, 0:2] B[0:2, 2:5] -> C[2:3, 2:5]
    # Block 8: A[2:3, 2:4] B[2:4, 2:5] -> C[2:3, 2:5]
    
    # This covers all of C.
    # Rank 59.
    
    # Convert to lists
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    return {"U": U, "V": V, "W": W}


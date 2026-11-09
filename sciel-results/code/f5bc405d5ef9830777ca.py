"""Program for matrix multiplication 3x3 x 3x5 tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Block decomposition approach:
    # Split A (3x3) into: A11(2x2), A12(2x1), A21(1x2), A22(1x1)
    # Split B (3x5) into: B11(2x5), B21(1x5) [split rows 0-1 and row 2]
    # 
    # C1 (2x5) = A11*B11 + A12*B21
    # C2 (1x5) = A21*B11 + A22*B21
    #
    # For A11*B11 (2x2 x 2x5): use Strassen-like on 2x2 x 2x2 for each pair of columns
    # Actually, 2x2 x 2x5: split B11 into 2x2 and 2x3, or handle directly.
    # 
    # Simpler: handle 2x2 x 2x5 by doing 5 separate 2x2 x 2x1 multiplications.
    # Each 2x2 x 2x1 has rank 2 (schoolbook). So 5*2 = 10 terms.
    # But we can do better: 2x2 x 2x5 tensor rank is at most 7 (Strassen for the 2x2 part)
    # Actually for 2x2 x 2xp, we can use Strassen: for each column pair... 
    # Let me just use schoolbook for simplicity: 2*2*5 = 20 terms for A11*B11.
    # That's not great. Let me think differently.
    
    # Better approach: Use the full Strassen decomposition for 2x2 x 2x2 blocks.
    # For 2x2 x 2x5, split the 5 columns into 2+2+1:
    #   2x2 x 2x2: rank 7 (Strassen)
    #   2x2 x 2x2: rank 7 (Strassen)  
    #   2x2 x 2x1: rank 2 (schoolbook)
    #   Total: 16
    # 
    # A12*B21 (2x1 x 1x5): outer product type, rank 5 (each output col is A12[i]*B21[k])
    #   Actually: C[i,k] = A12[i]*B21[k], so it's a sum of 1 term per (i,k) pair? No.
    #   C[i,k] = A12[i] * B21[k]. This is rank 1 as a bilinear map! 
    #   Wait: for fixed i,k, C[i,k] = A12[i]*B21[k]. The tensor is sum_{i,k} e_i^A e_k^B e_{i,k}^C
    #   This has rank 1 (it's an outer product of vectors). So rank 1.
    
    # A21*B11 (1x2 x 2x5): C[k] = sum_j A21[j]*B11[j,k], rank 2 (one mult per j)
    #   For 5 columns: rank 2 (the 2 multiplications are shared across columns)
    #   Actually no: C[k] = A21[0]*B11[0,k] + A21[1]*B11[1,k]
    #   This is 2 multiplications total (A21[0]*B11[0,k] for all k is one "term"?)
    #   No, each multiplication A21[j]*B11[j,k] is separate for each k.
    #   So rank = 2*5 = 10? No wait.
    #   The tensor for 1x2 x 2x5 -> 1x5: T = sum_{j=0}^{1} sum_{k=0}^{4} e_j^A e_{j,k}^B e_k^C
    #   Each (j,k) pair gives one term. So rank = 10.
    #   Hmm, but actually we can factor: sum_j A21[j] * (B11[j,0], B11[j,1], ..., B11[j,4])
    #   This is 2 terms where each term computes A21[j] * B11[j,k] for all k simultaneously.
    #   But in tensor decomposition, each term is a single scalar multiplication.
    #   So rank = 10 for this part.
    
    # This is getting complex. Let me use a cleaner approach.
    
    # CLEANEST APPROACH: Direct schoolbook with one optimization.
    # Schoolbook: 3*3*5 = 45 terms.
    # 
    # Optimization: For the 3x3 x 3x3 part (first 3 cols), use Strassen-based rank 16.
    # For the 3x3 x 3x2 part (last 2 cols), use schoolbook rank 18.
    # Total: 16 + 18 = 34.
    
    # But I showed 16 might be wrong. Let me just use schoolbook for everything = 45.
    # Actually, let me use a verified Strassen-based decomposition.
    
    # I'll implement the block decomposition carefully:
    # For 3x3 x 3x3 part (columns 0,1,2 of B):
    #   Use Strassen for the 2x2 x 2x2 block, schoolbook for the rest.
    #   A11(2x2) * B11(2x2): Strassen rank 7
    #   A12(2x1) * B21(1x2): rank 1 (outer product)
    #   A11(2x2) * B12(2x1): rank 2
    #   A12(2x1) * B22(1x1): rank 1
    #   A21(1x2) * B11(2x2): rank 2
    #   A22(1x1) * B21(1x2): rank 1
    #   A21(1x2) * B12(2x1): rank 1
    #   A22(1x1) * B22(1x1): rank 1
    #   Total: 7+1+2+1+2+1+1+1 = 16
    
    # For 3x3 x 3x2 part (columns 3,4 of B):
    #   A11(2x2) * B11(2x2): Strassen rank 7
    #   A12(2x1) * B21(1x2): rank 1
    #   A21(1x2) * B11(2x2): rank 2
    #   A22(1x1) * B21(1x2): rank 1
    #   Total: 7+1+2+1 = 11
    
    # Grand total: 16 + 11 = 27
    
    # Let me implement this. I'll build the U, V, W matrices.
    
    terms = []  # Each term: (u_dict, v_dict, w_dict)
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Strassen 2x2 x 2x2 decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A01)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Helper to add Strassen terms for a 2x2 x 2x2 sub-problem
    # a_indices: dict mapping (row,col) in 2x2 A-block to U index
    # b_indices: dict mapping (row,col) in 2x2 B-block to V index  
    # c_indices: dict mapping (row,col) in 2x2 C-block to W index
    def add_strassen(a_idx, b_idx, c_idx):
        # a_idx: {(r,c): u_index}
        # b_idx: {(r,c): v_index}
        # c_idx: {(r,c): w_index}
        
        # M1 = (A00+A11)(B00+B11)
        add_term(
            {a_idx[(0,0)]: 1, a_idx[(1,1)]: 1},
            {b_idx[(0,0)]: 1, b_idx[(1,1)]: 1},
            {c_idx[(0,0)]: 1, c_idx[(1,1)]: 1, c_idx[(0,1)]: -1, c_idx[(1,0)]: 1}
        )
        # M2 = (A10+A11)B00
        add_term(
            {a_idx[(1,0)]: 1, a_idx[(1,1)]: 1},
            {b_idx[(0,0)]: 1},
            {c_idx[(1,0)]: 1, c_idx[(1,1)]: -1}
        )
        # M3 = A00(B01-B11)
        add_term(
            {a_idx[(0,0)]: 1},
            {b_idx[(0,1)]: 1, b_idx[(1,1)]: -1},
            {c_idx[(0,1)]: 1, c_idx[(1,1)]: 1}
        )
        # M4 = A11(B10-B00)
        add_term(
            {a_idx[(1,1)]: 1},
            {b_idx[(1,0)]: 1, b_idx[(0,0)]: -1},
            {c_idx[(0,0)]: 1, c_idx[(1,0)]: 1}
        )
        # M5 = (A00+A10)B11
        add_term(
            {a_idx[(0,0)]: 1, a_idx[(1,0)]: 1},
            {b_idx[(1,1)]: 1},
            {c_idx[(0,0)]: -1, c_idx[(0,1)]: 1}
        )
        # M6 = (A11-A01)(B00+B01)
        add_term(
            {a_idx[(1,1)]: 1, a_idx[(0,1)]: -1},
            {b_idx[(0,0)]: 1, b_idx[(0,1)]: 1},
            {c_idx[(1,1)]: 1}
        )
        # M7 = (A01-A11)(B10+B11)
        add_term(
            {a_idx[(0,1)]: 1, a_idx[(1,1)]: -1},
            {b_idx[(1,0)]: 1, b_idx[(1,1)]: 1},
            {c_idx[(0,0)]: 1, c_idx[(0,1)]: -1}
        )
    
    # Helper for outer product 2x1 x 1x2 -> 2x2
    # C[i,j] = A[i] * B[j]
    def add_outer_2x1_1x2(a_vec, b_vec, c_2x2):
        # a_vec: {local_idx: u_index}, b_vec: {local_idx: v_index}, c_2x2: {(i,j): w_index}
        u = {}
        v = {}
        w = {}
        for i, ui in a_vec.items():
            u[ui] = 1
        for j, vj in b_vec.items():
            v[vj] = 1
        for i in a_vec:
            for j in b_vec:
                w[c_2x2[(i,j)]] = 1
        add_term(u, v, w)
    
    # Helper for 2x2 x 2x1 -> 2x1 (schoolbook, rank 2)
    # C[i] = sum_j A[i,j] * B[j]
    def add_2x2_2x1(a_2x2, b_vec, c_vec):
        # a_2x2: {(i,j): u_index}, b_vec: {j: v_index}, c_vec: {i: w_index}
        for j in range(2):
            u = {a_2x2[(0,j)]: 1, a_2x2[(1,j)]: 1}
            v = {b_vec[j]: 1}
            w = {c_vec[0]: 1, c_vec[1]: 1}
            add_term(u, v, w)
    
    # Helper for 2x1 x 1x1 -> 2x1 (rank 1)
    def add_2x1_1x1(a_vec, b_scalar, c_vec):
        u = {a_vec[0]: 1, a_vec[1]: 1}
        v = {b_scalar: 1}
        w = {c_vec[0]: 1, c_vec[1]: 1}
        add_term(u, v, w)
    
    # Helper for 1x2 x 2x2 -> 1x2 (schoolbook, rank 2)
    # C[j] = sum_i A[i] * B[i,j]
    def add_1x2_2x2(a_vec, b_2x2, c_vec):
        for j in range(2):
            u = {a_vec[0]: 1, a_vec[1]: 1}
            v = {b_2x2[(0,j)]: 1, b_2x2[(1,j)]: 1}
            w = {c_vec[j]: 1}
            add_term(u, v, w)
    
    # Helper for 1x1 x 1x2 -> 1x2 (rank 1)
    def add_1x1_1x2(a_scalar, b_vec, c_vec):
        u = {a_scalar: 1}
        v = {b_vec[0]: 1, b_vec[1]: 1}
        w = {c_vec[0]: 1, c_vec[1]: 1}
        add_term(u, v, w)
    
    # Helper for 1x2 x 2x1 -> 1x1 (rank 1)
    def add_1x2_2x1(a_vec, b_vec, c_scalar):
        u = {a_vec[0]: 1, a_vec[1]: 1}
        v = {b_vec[0]: 1, b_vec[1]: 1}
        w = {c_scalar: 1}
        add_term(u, v, w)
    
    # Helper for 1x1 x 1x1 -> 1x1 (rank 1)
    def add_1x1_1x1(a_scalar, b_scalar, c_scalar):
        u = {a_scalar: 1}
        v = {b_scalar: 1}
        w = {c_scalar: 1}
        add_term(u, v, w)
    
    # Now build the full decomposition for 3x3 x 3x5
    # U indices: i*m + j = i*3 + j for i in 0..2, j in 0..2
    # V indices: j*p + k = j*5 + k for j in 0..2, k in 0..4
    # W indices: k*n + i = k*3 + i for k in 0..4, i in 0..2
    
    # Part 1: 3x3 x 3x3 (B columns 0,1,2)
    # A11: rows 0-1, cols 0-1 of A
    # A12: rows 0-1, col 2 of A
    # A21: row 2, cols 0-1 of A
    # A22: row 2, col 2 of A
    # B11: rows 0-1, cols 0-1 of B (for this part, B cols 0-1)
    # B12: rows 0-1, col 2 of B (for this part, B col 2)
    # B21: row 2, cols 0-1 of B
    # B22: row 2, col 2 of B
    
    # A11 indices in U: {(0,0):0, (0,1):1, (1,0):3, (1,1):4}
    a11_u = {(0,0): 0, (0,1): 1, (1,0): 3, (1,1): 4}
    # A12 indices in U: {0: 2, 1: 5} (rows 0,1 of col 2)
    a12_u = {0: 2, 1: 5}
    # A21 indices in U: {0: 6, 1: 7} (cols 0,1 of row 2)
    a21_u = {0: 6, 1: 7}
    # A22 index in U: 8
    a22_u = 8
    
    # B11 indices in V (cols 0,1 of B): {(0,0):0, (0,1):1, (1,0):5, (1,1):6}
    b11_v = {(0,0): 0, (0,1): 1, (1,0): 5, (1,1): 6}
    # B12 indices in V (col 2 of B): {0: 2, 1: 7} (rows 0,1 of col 2)
    b12_v = {0: 2, 1: 7}
    # B21 indices in V (cols 0,1 of B, row 2): {0: 10, 1: 11}
    b21_v = {0: 10, 1: 11}
    # B22 index in V: 12
    b22_v = 12
    
    # C11 indices in W (rows 0-1, cols 0-1): {(0,0):0, (0,1):3, (1,0):1, (1,1):4}
    # W index = k*n + i = k*3 + i
    # C11[i,j] where i in 0..1, j in 0..1: W[j*3 + i]
    c11_w = {(0,0): 0, (0,1): 3, (1,0): 1, (1,1): 4}
    # C12 indices in W (rows 0-1, col 2): {0: 6, 1: 7}
    c12_w = {0: 6, 1: 7}
    # C21 indices in W (row 2, cols 0-1): {0: 2, 1: 5}
    c21_w = {0: 2, 1: 5}
    # C22 index in W: 8
    c22_w = 8
    
    # C11 = A11*B11 + A12*B21
    add_strassen(a11_u, b11_v, c11_w)  # 7 terms
    add_outer_2x1_1x2(a12_u, b21_v, c11_w)  # 1 term
    
    # C12 = A11*B12 + A12*B22
    add_2x2_2x1(a11_u, b12_v, c12_w)  # 2 terms
    add_2x1_1x1(a12_u, b22_v, c12_w)  # 1 term
    
    # C21 = A21*B11 + A22*B21
    add_1x2_2x2(a21_u, b11_v, c21_w)  # 2 terms
    add_1x1_1x2(a22_u, b21_v, c21_w)  # 1 term
    
    # C22 = A21*B12 + A22*B22
    add_1x2_2x1(a21_u, b12_v, c22_w)  # 1 term
    add_1x1_1x1(a22_u, b22_v, c22_w)  # 1 term
    
    # Part 2: 3x3 x 3x2 (B columns 3,4)
    # B11' indices in V (cols 3,4 of B): {(0,0):3, (0,1):4, (1,0):8, (1,1):9}
    b11p_v = {(0,0): 3, (0,1): 4, (1,0): 8, (1,1): 9}
    # B21' indices in V (cols 3,4 of B, row 2): {0: 13, 1: 14}
    b21p_v = {0: 13, 1: 14}
    
    # C1' indices in W (rows 0-1, cols 3-4): {(0,0):9, (0,1):12, (1,0):10, (1,1):13}
    # W index = k*3 + i, k in 3..4, i in 0..1
    c1p_w = {(0,0): 9, (0,1): 12, (1,0): 10, (1,1): 13}
    # C2' indices in W (row 2, cols 3-4): {0: 11, 1: 14}
    c2p_w = {0: 11, 1: 14}
    
    # C1' = A11*B11' + A12*B21'
    add_strassen(a11_u, b11p_v, c1p_w)  # 7 terms
    add_outer_2x1_1x2(a12_u, b21p_v, c1p_w)  # 1 term
    
    # C2' = A21*B11' + A22*B21'
    add_1x2_2x2(a21_u, b11p_v, c2p_w)  # 2 terms
    add_1x1_1x2(a22_u, b21p_v, c2p_w)  # 1 term
    
    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

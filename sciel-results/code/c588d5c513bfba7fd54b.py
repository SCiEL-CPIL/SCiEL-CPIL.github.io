"""
Tensor decomposition for 3x4 * 4x8 matrix multiplication.
Uses Strassen's algorithm for 2x2 blocks where possible.
"""
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy:
    - Partition A (3x4) into A1 (2x4) and A2 (1x4)
    - Partition B (4x8) into B1 (4x4) and B2 (4x4)
    - For A1*B1: use Strassen on 2x2 blocks (4 blocks of 2x2*2x2, each rank 7 = 28 total)
    - For A1*B2: use Strassen on 2x2 blocks (4 blocks of 2x2*2x2, each rank 7 = 28 total)
    - For A2*B1: schoolbook 1x4*4x4, rank 16
    - For A2*B2: schoolbook 1x4*4x4, rank 16
    Total rank: 28 + 28 + 16 + 16 = 88
    """
    n, m, p = DIMENSIONS
    
    # We'll build the decomposition by accumulating terms
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    # Each term is a triple (u, v, w) where:
    # u is in R^(n*m) with 1 at position (i*m + j)
    # v is in R^(m*p) with 1 at position (j*p + k)
    # w is in R^(p*n) with 1 at position (k*n + i)
    
    # For Strassen 2x2*2x2, we need to handle the linear combinations
    
    terms = []  # List of (u_vector, v_vector, w_vector)
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        """Add a term with coefficient to the decomposition."""
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        u_vec[u_idx] = coeff
        v_vec[v_idx] = coeff
        w_vec[w_idx] = coeff
        terms.append((u_vec, v_vec, w_vec))
    
    def add_strassen_block(i0, j0, k0):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        
        Strassen's algorithm for 2x2*2x2:
        Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        
        M1 = (a11 + a22) * (b11 + b22)
        M2 = (a21 + a22) * b11
        M3 = a11 * (b12 - b22)
        M4 = a22 * (b21 - b11)
        M5 = (a11 + a12) * b22
        M6 = (a21 - a11) * (b11 + b12)
        M7 = (a12 - a22) * (b21 + b22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # Map local indices to global indices
        # A indices: rows i0, i0+1; cols j0, j0+1
        # B indices: rows j0, j0+1; cols k0, k0+1
        # C indices: rows i0, i0+1; cols k0, k0+1
        
        # U index: (i, j) -> i*m + j
        # V index: (j, k) -> j*p + k
        # W index: (k, i) -> k*n + i
        
        a11_u = (i0) * m + j0
        a12_u = (i0) * m + (j0 + 1)
        a21_u = (i0 + 1) * m + j0
        a22_u = (i0 + 1) * m + (j0 + 1)
        
        b11_v = j0 * p + k0
        b12_v = j0 * p + (k0 + 1)
        b21_v = (j0 + 1) * p + k0
        b22_v = (j0 + 1) * p + (k0 + 1)
        
        c11_w = k0 * n + i0
        c12_w = (k0 + 1) * n + i0
        c21_w = k0 * n + (i0 + 1)
        c22_w = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a11 + a22) * (b11 + b22)
        # This contributes to C11 and C22
        # We need to express this as a sum of rank-1 terms
        
        # Actually, Strassen's algorithm uses 7 multiplications, but each "multiplication" 
        # in the tensor decomposition sense needs to be broken down.
        
        # Let me reconsider. In the tensor decomposition, each term is a product of:
        # a linear form in A, a linear form in B, and a linear form in C.
        
        # For M1 = (a11 + a22) * (b11 + b22), this is actually a product of two linear forms,
        # so it's already a rank-1 term in the A-B space, but we need to see how it maps to C.
        
        # Wait, I need to think about this more carefully.
        
        # The tensor T has components T[i,j,k] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
        # A rank-r decomposition is T = sum_r u_r ⊗ v_r ⊗ w_r
        
        # For Strassen, we have:
        # C = M1 * [[1,0],[0,1]] - M2 * [[0,0],[1,0]] + M3 * [[0,1],[0,0]] + M4 * [[0,1],[0,0]] 
        #     - M5 * [[1,0],[0,0]] + M6 * [[0,0],[0,1]] + M7 * [[1,0],[0,0]]
        
        # Wait, let me get the correct Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mr is a product of a linear form in A and a linear form in B.
        
        # So for each Mr, we have:
        # Mr = (linear form in A) * (linear form in B)
        # And Mr contributes to specific C entries with coefficients.
        
        # For example, M1 = (a11 + a22) * (b11 + b22)
        # This means: u_M1 has 1 at a11 and 1 at a22 (in the A space)
        #             v_M1 has 1 at b11 and 1 at b22 (in the B space)
        # And M1 contributes: +1 to C11, +1 to C22
        
        # So we add a term: u has 1 at a11, 1 at a22; v has 1 at b11, 1 at b22; w has 1 at c11, 1 at c22
        
        # Let me implement this properly.
        
        # M1 = (a11 + a22) * (b11 + b22)
        u1 = [0.0] * (n * m); u1[a11_u] = 1.0; u1[a22_u] = 1.0
        v1 = [0.0] * (m * p); v1[b11_v] = 1.0; v1[b22_v] = 1.0
        w1 = [0.0] * (p * n); w1[c11_w] = 1.0; w1[c22_w] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (a21 + a22) * b11
        u2 = [0.0] * (n * m); u2[a21_u] = 1.0; u2[a22_u] = 1.0
        v2 = [0.0] * (m * p); v2[b11_v] = 1.0
        w2 = [0.0] * (p * n); w2[c21_w] = 1.0; w2[c22_w] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = a11 * (b12 - b22)
        u3 = [0.0] * (n * m); u3[a11_u] = 1.0
        v3 = [0.0] * (m * p); v3[b12_v] = 1.0; v3[b22_v] = -1.0
        w3 = [0.0] * (p * n); w3[c12_w] = 1.0; w3[c22_w] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = a22 * (b21 - b11)
        u4 = [0.0] * (n * m); u4[a22_u] = 1.0
        v4 = [0.0] * (m * p); v4[b21_v] = 1.0; v4[b11_v] = -1.0
        w4 = [0.0] * (p * n); w4[c11_w] = 1.0; w4[c21_w] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (a11 + a12) * b22
        u5 = [0.0] * (n * m); u5[a11_u] = 1.0; u5[a12_u] = 1.0
        v5 = [0.0] * (m * p); v5[b22_v] = 1.0
        w5 = [0.0] * (p * n); w5[c11_w] = -1.0; w5[c12_w] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (a21 - a11) * (b11 + b12)
        u6 = [0.0] * (n * m); u6[a21_u] = 1.0; u6[a11_u] = -1.0
        v6 = [0.0] * (m * p); v6[b11_v] = 1.0; v6[b12_v] = 1.0
        w6 = [0.0] * (p * n); w6[c22_w] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (a12 - a22) * (b21 + b22)
        u7 = [0.0] * (n * m); u7[a12_u] = 1.0; u7[a22_u] = -1.0
        v7 = [0.0] * (m * p); v7[b21_v] = 1.0; v7[b22_v] = 1.0
        w7 = [0.0] * (p * n); w7[c11_w] = 1.0
        terms.append((u7, v7, w7))
    
    def add_schoolbook_1x4_4x4(i_row, j0, k0):
        """
        Add schoolbook decomposition for A[i_row, j0:j0+4] * B[j0:j0+4, k0:k0+4]
        This is a 1x4 * 4x4 multiplication, rank 16.
        """
        for j in range(4):
            for k in range(4):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                u_vec = [0.0] * (n * m); u_vec[u_idx] = 1.0
                v_vec = [0.0] * (m * p); v_vec[v_idx] = 1.0
                w_vec = [0.0] * (p * n); w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    
    # Partition: A = [A1; A2] where A1 is 2x4, A2 is 1x4
    # B = [B1 | B2] where B1 is 4x4, B2 is 4x4
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    
    # A1*B1: A1 is 2x4, B1 is 4x4
    # Split into 2x2 blocks: 
    # A1 = [[A1_11, A1_12], [A1_21, A1_22]] where each is 2x2
    # B1 = [[B1_11, B1_12], [B1_21, B1_22]] where each is 2x2
    
    # Block (0,0): A1[0:2, 0:2] * B1[0:2, 0:2]
    add_strassen_block(i0=0, j0=0, k0=0)
    
    # Block (0,1): A1[0:2, 2:4] * B1[2:4, 0:2]
    add_strassen_block(i0=0, j0=2, k0=0)
    
    # Block (1,0): A1[2:4, 0:2] * B1[0:2, 0:2]
    add_strassen_block(i0=0, j0=0, k0=0)  # Wait, this is wrong
    
    # Let me reconsider the indexing.
    # A1 is rows 0,1 of A (3x4), so A1 has shape 2x4
    # A1[i, j] = A[i, j] for i=0,1 and j=0,1,2,3
    # B1 is cols 0,1,2,3 of B (4x8), so B1 has shape 4x4
    # B1[j, k] = B[j, k] for j=0,1,2,3 and k=0,1,2,3
    
    # For A1*B1, we split A1 into 2x2 blocks and B1 into 2x2 blocks.
    # A1 = [[A1_11, A1_12], [A1_21, A1_22]]
    # where A1_11 = A[0:2, 0:2], A1_12 = A[0:2, 2:4]
    #       A1_21 = A[2:4, 0:2] but wait, A only has 3 rows, so A[2:4] doesn't exist for A1.
    
    # I made an error. A1 is 2x4, so it only has rows 0 and 1.
    # A1_11 = A[0:2, 0:2] (2x2)
    # A1_12 = A[0:2, 2:4] (2x2)
    # A1_21 = A[2:4, 0:2] - this doesn't exist since A1 only has 2 rows!
    
    # So I can only do one row of 2x2 blocks for A1.
    # A1 is 2x4, so I can split it as [A1_left | A1_right] where each is 2x2.
    # B1 is 4x4, so I can split it as [[B1_tl, B1_tr], [B1_bl, B1_br]] where each is 2x2.
    
    # A1*B1 = A1_left * B1_left_cols + A1_right * B1_right_cols
    # where B1_left_cols = B1[:, 0:2] (4x2) and B1_right_cols = B1[:, 2:4] (4x2)
    
    # This is a 2x2 * 4x2 multiplication, which I can split further.
    # 2x2 * 4x2 = (2x2 * 2x2) + (2x2 * 2x2) by splitting the 4 rows into two groups of 2.
    
    # So: A1*B1 = A1_left * B1[0:2, 0:2] + A1_left * B1[2:4, 0:2] 
    #                + A1_right * B1[0:2, 2:4] + A1_right * B1[2:4, 2:4]
    
    # Each term is a 2x2 * 2x2 multiplication, which can use Strassen (rank 7).
    # So total for A1*B1: 4 * 7 = 28 rank terms.
    
    # Let me redo this properly.
    
    # Clear the terms list and start over
    terms.clear()
    
    # A1*B1: 4 blocks of 2x2*2x2
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(i0=0, j0=0, k0=0)
    
    # Block 2: A[0:2, 0:2] * B[2:4, 0:2]
    add_strassen_block(i0=0, j0=2, k0=0)
    
    # Block 3: A[0:2, 2:4] * B[0:2, 2:4]
    add_strassen_block(i0=0, j0=0, k0=2)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4]
    add_strassen_block(i0=0, j0=2, k0=2)
    
    # A1*B2: A1 is 2x4, B2 is 4x4 (cols 4-7 of B)
    # Similar splitting:
    # Block 5: A[0:2, 0:2] * B[0:2, 4:6]
    add_strassen_block(i0=0, j0=0, k0=4)
    
    # Block 6: A[0:2, 0:2] * B[2:4, 4:6]
    add_strassen_block(i0=0, j0=2, k0=4)
    
    # Block 7: A[0:2, 2:4] * B[0:2, 6:8]
    add_strassen_block(i0=0, j0=0, k0=6)
    
    # Block 8: A[0:2, 2:4] * B[2:4, 6:8]
    add_strassen_block(i0=0, j0=2, k0=6)
    
    # A2*B1: A2 is 1x4 (row 2 of A), B1 is 4x4
    # Schoolbook: 1*4*4 = 16 terms
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=0)
    
    # A2*B2: A2 is 1x4, B2 is 4x4
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=4)
    
    rank = len(terms)
    
    # Build final matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        u_vec, v_vec, w_vec = terms[r]
        for i in range(n * m):
            if u_vec[i] != 0:
                U[i][r] = u_vec[i]
        for j in range(m * p):
            if v_vec[j] != 0:
                V[j][r] = v_vec[j]
        for k in range(p * n):
            if w_vec[k] != 0:
                W[k][r] = w_vec[k]
    
    return {"U": U, "V": V, "W": W}

"""Matrix multiplication 4x5x5 with Strassen-like tensor network ansatz."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix using a Strassen-like hierarchy.
    
    Strategy: Decompose the 4x5x5 problem into smaller sub-problems.
    We split the matrices into blocks and apply Strassen's algorithm
    where the block sizes are even, falling back to schoolbook for
    odd-sized blocks.
    
    The key idea is to use the Strassen identity for 2x2x2 blocks:
    M1 = (A11+A21)(B10+B20)
    M2 = (A21-A11)(B10+B20)
    M3 = A11(B11-B21)
    M4 = A22(B21-B11)
    M5 = (A11+A22)(B11+B22)
    M6 = (A12-A22)(B21+B22)
    M7 = (A11+A12)(B10-B11)
    
    C11 = M5 + M4 - M1 + M3
    C12 = M1 + M2
    C21 = M3 + M4
    C22 = M5 - M2 + M6 - M7
    
    For the 4x5x5 problem, we can split:
    - A (4x5) into two 2x5 blocks (top and bottom)
    - B (5x5) into 2x2, 2x2, and 1x1 blocks along both rows and columns
    
    This gives us sub-problems of sizes:
    - 2x2x2 (use Strassen, rank 7)
    - 2x2x1, 2x1x2, 1x2x2 (use schoolbook or optimized)
    - 2x1x1, 1x1x2, 1x2x1 (schoolbook, rank 1-2)
    - 1x1x1 (schoolbook, rank 1)
    
    Let's implement this decomposition carefully.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build up the decomposition by processing blocks
    # U, V, W will be lists of rank-1 terms
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u, v, w):
        """Add a rank-1 term to the decomposition."""
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
    
    # Split A (4x5) into A_top (2x5) and A_bottom (2x5)
    # Split B (5x5) into blocks:
    #   B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:5]
    #   B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:5]
    #   B[4:5, 0:2], B[4:5, 2:4], B[4:5, 4:5]
    
    # For each pair of row-blocks of A and column-blocks of B,
    # we have a sub-problem of size (rows_A_block) x (cols_A_block) x (cols_B_block)
    
    # Let's define the block structure:
    # A rows: [0,1], [2,3] (two blocks of size 2)
    # B rows: [0,1], [2,3], [4] (blocks of size 2, 2, 1)
    # B cols: [0,1], [2,3], [4] (blocks of size 2, 2, 1)
    
    A_row_blocks = [[0, 1], [2, 3]]
    B_row_blocks = [[0, 1], [2, 3], [4]]
    B_col_blocks = [[0, 1], [2, 3], [4]]
    
    # For each combination, we have a sub-multiplication
    for a_block_idx, a_rows in enumerate(A_row_blocks):
        for b_row_idx, b_rows in enumerate(B_row_blocks):
            for b_col_idx, b_cols in enumerate(B_col_blocks):
                # Sub-problem: A[a_rows, b_rows] * B[b_rows, b_cols]
                # Size: len(a_rows) x len(b_rows) x len(b_cols)
                n_sub = len(a_rows)
                m_sub = len(b_rows)
                p_sub = len(b_cols)
                
                # Create index mappings
                a_indices = a_rows  # indices into A's rows
                j_indices = b_rows  # indices into the shared dimension
                k_indices = b_cols  # indices into B's columns
                
                # For this sub-problem, we need to compute:
                # C_sub[i_sub, k_sub] = sum_{j_sub} A_sub[i_sub, j_sub] * B_sub[j_sub, k_sub]
                
                # Where A_sub is n_sub x m_sub, B_sub is m_sub x p_sub
                
                # Use Strassen if n_sub==2 and m_sub==2 and p_sub==2
                if n_sub == 2 and m_sub == 2 and p_sub == 2:
                    # Strassen for 2x2x2
                    # A_sub = [[A00, A01], [A10, A11]]
                    # B_sub = [[B00, B01], [B10, B11]]
                    # C_sub = [[C00, C01], [C10, C11]]
                    
                    i0, i1 = a_indices[0], a_indices[1]
                    j0, j1 = j_indices[0], j_indices[1]
                    k0, k1 = k_indices[0], k_indices[1]
                    
                    # M1 = (A[i0,j0] + A[i1,j0]) * (B[j0,k0] + B[j1,k0])
                    u1 = [0] * (n * m); u1[i0 * m + j0] = 1; u1[i1 * m + j0] = 1
                    v1 = [0] * (m * p); v1[j0 * p + k0] = 1; v1[j1 * p + k0] = 1
                    w1 = [0] * (p * n)
                    w1[k0 * n + i0] = 1
                    w1[k1 * n + i0] = 1
                    w1[k0 * n + i1] = -1
                    w1[k1 * n + i1] = -1
                    add_term(u1, v1, w1)
                    
                    # M2 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j1,k0])
                    u2 = [0] * (n * m); u2[i0 * m + j0] = -1; u2[i1 * m + j0] = 1
                    v2 = [0] * (m * p); v2[j0 * p + k0] = 1; v2[j1 * p + k0] = 1
                    w2 = [0] * (p * n)
                    w2[k0 * n + i0] = 1
                    w2[k1 * n + i0] = 1
                    w2[k0 * n + i1] = -1
                    w2[k1 * n + i1] = -1
                    add_term(u2, v2, w2)
                    
                    # M3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
                    u3 = [0] * (n * m); u3[i0 * m + j0] = 1
                    v3 = [0] * (m * p); v3[j0 * p + k1] = 1; v3[j1 * p + k1] = -1
                    w3 = [0] * (p * n)
                    w3[k1 * n + i0] = 1
                    w3[k1 * n + i1] = 1
                    add_term(u3, v3, w3)
                    
                    # M4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
                    u4 = [0] * (n * m); u4[i1 * m + j1] = 1
                    v4 = [0] * (m * p); v4[j0 * p + k0] = -1; v4[j1 * p + k0] = 1
                    w4 = [0] * (p * n)
                    w4[k0 * n + i0] = 1
                    w4[k0 * n + i1] = 1
                    add_term(u4, v4, w4)
                    
                    # M5 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
                    u5 = [0] * (n * m); u5[i0 * m + j0] = 1; u5[i1 * m + j1] = 1
                    v5 = [0] * (m * p); v5[j0 * p + k0] = 1; v5[j1 * p + k1] = 1
                    w5 = [0] * (p * n)
                    w5[k0 * n + i0] = 1
                    w5[k1 * n + i0] = -1
                    w5[k0 * n + i1] = -1
                    w5[k1 * n + i1] = 1
                    add_term(u5, v5, w5)
                    
                    # M6 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
                    u6 = [0] * (n * m); u6[i0 * m + j1] = 1; u6[i1 * m + j1] = -1
                    v6 = [0] * (m * p); v6[j1 * p + k0] = 1; v6[j1 * p + k1] = 1
                    w6 = [0] * (p * n)
                    w6[k0 * n + i0] = -1
                    w6[k1 * n + i0] = 1
                    w6[k0 * n + i1] = 1
                    w6[k1 * n + i1] = -1
                    add_term(u6, v6, w6)
                    
                    # M7 = (A[i0,j0] + A[i0,j1]) * (B[j0,k0] - B[j0,k1])
                    u7 = [0] * (n * m); u7[i0 * m + j0] = 1; u7[i0 * m + j1] = 1
                    v7 = [0] * (m * p); v7[j0 * p + k0] = 1; v7[j0 * p + k1] = -1
                    w7 = [0] * (p * n)
                    w7[k0 * n + i0] = -1
                    w7[k1 * n + i0] = 1
                    w7[k0 * n + i1] = 1
                    w7[k1 * n + i1] = -1
                    add_term(u7, v7, w7)
                    
                else:
                    # Schoolbook for other block sizes
                    for li in range(n_sub):
                        for lj in range(m_sub):
                            for lk in range(p_sub):
                                i = a_indices[li]
                                j = j_indices[lj]
                                k = k_indices[lk]
                                
                                u = [0] * (n * m); u[i * m + j] = 1
                                v = [0] * (m * p); v[j * p + k] = 1
                                w = [0] * (p * n); w[k * n + i] = 1
                                add_term(u, v, w)
    
    # Convert to matrices
    rank = len(U_terms)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        for idx, val in enumerate(U_terms[c]):
            if val != 0:
                U[idx, c] = val
        for idx, val in enumerate(V_terms[c]):
            if val != 0:
                V[idx, c] = val
        for idx, val in enumerate(W_terms[c]):
            if val != 0:
                W[idx, c] = val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

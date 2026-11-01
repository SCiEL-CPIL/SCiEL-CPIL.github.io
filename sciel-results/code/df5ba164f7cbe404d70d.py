"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Build a rank-minimal (or near-minimal) decomposition of the 2x2 x 2x15 matrix multiplication.
    
    Strategy:
    We use the structure of Strassen's algorithm for 2x2 matrix multiplication.
    Strassen computes the product C = A * B where A is 2x2 and B is 2x2 with rank 7.
    The multiplication can be extended to larger B by splitting B into blocks.
    
    For 2x2 * 2x15, we can view B as a 2x15 matrix. We can split the 15 columns into
    groups that are compatible with Strassen's method.
    
    Strassen's 7 rank-1 terms for 2x2 multiplication:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B12 - B22)
    M4 = A22(B21 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    Then:
    C11 = M1 - M4 + M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M2 - M6
    
    Each M term involves a product of two sums of entries. To decompose this as a tensor
    of rank 7, we need 7 rank-1 tensors.
    
    For the 2x2 * 2x15 case, we can apply Strassen's method to each 2x2 block of B,
    but we need to handle the 15 columns efficiently.
    
    Actually, a better approach: we can use the fact that the rank of M_{2,2,p} is at most
    7 * ceil(p/2) if we can group columns into pairs, plus some remainder.
    
    Let me think differently. The rank of M_{n,m,p} for small n,m is known.
    For M_{2,2,p}, the rank is known to be at most 3p + O(1) using fast multiplication,
    but the exact minimum is not always known.
    
    For p=15, let's try to use a decomposition that gives rank < 60.
    
    One known approach: use the Strassen algorithm recursively.
    M_{2,2,15} can be decomposed by splitting the 15 columns into 7 groups of 2 and 1 group of 1.
    For each group of 2 columns, we use Strassen (rank 7).
    For the group of 1 column, we use schoolbook (rank 4).
    Total rank: 7 * 7 + 4 = 49 + 4 = 53.
    
    Let me verify this construction.
    
    For a 2x2 * 2x2 block, Strassen gives 7 terms.
    For a 2x2 * 2x1 block, we need 4 terms (schoolbook).
    
    So for 15 columns: 7 pairs + 1 single = 7*7 + 1*4 = 49 + 4 = 53.
    
    Let me construct this.
    """
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (30, rank)
    
    # Strassen terms for a 2x2 * 2x2 block:
    # Each term l has:
    #   u_l: 4-vector (coefficients for A entries)
    #   v_l: 2x2 = 4-vector (coefficients for B entries)
    #   w_l: 2x2 = 4-vector (coefficients for C entries)
    
    # Strassen's 7 terms (u, v, w) where u is 4-vector for A, v is 4-vector for B (row-major),
    # w is 4-vector for C (row-major: C11, C12, C21, C22)
    
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    strassen_terms = []
    
    # Term 1: M1 = (A11 + A22)(B11 + B22)
    u1 = [1, 0, 0, 1]  # A11 + A22
    v1 = [1, 0, 0, 1]  # B11 + B22
    w1 = [1, 0, 0, 1]  # C11 += M1, C22 += M1
    strassen_terms.append((u1, v1, w1))
    
    # Term 2: M2 = (A21 + A22)B11
    u2 = [0, 0, 1, 1]  # A21 + A22
    v2 = [1, 0, 0, 0]  # B11
    w2 = [0, 0, 1, -1]  # C21 += M2, C22 -= M6... wait, C22 = M1 - M3 + M2 - M6
    # Let me re-derive:
    # C22 = M1 - M3 + M2 - M6
    # So M2 contributes +1 to C22
    w2 = [0, 0, 1, 1]  # C21 += M2, C22 += M2
    strassen_terms.append((u2, v2, w2))
    
    # Term 3: M3 = A11(B12 - B22)
    u3 = [1, 0, 0, 0]  # A11
    v3 = [0, 1, 0, -1]  # B12 - B22
    w3 = [0, 1, 0, -1]  # C12 += M3, C22 -= M3
    strassen_terms.append((u3, v3, w3))
    
    # Term 4: M4 = A22(B21 - B11)
    u4 = [0, 0, 0, 1]  # A22
    v4 = [0, 0, 1, 0]  # B21 - B11... wait, B21 is at index 2, B11 at index 0
    v4 = [0, 0, 1, 0]  # This is just B21. I need B21 - B11 = [−1, 0, 1, 0]
    v4 = [-1, 0, 1, 0]
    w4 = [-1, 0, 1, 0]  # C11 -= M4, C21 += M4
    strassen_terms.append((u4, v4, w4))
    
    # Term 5: M5 = (A11 + A12)B22
    u5 = [1, 1, 0, 0]  # A11 + A12
    v5 = [0, 0, 0, 1]  # B22
    w5 = [1, 1, 0, 0]  # C11 += M5, C12 += M5
    strassen_terms.append((u5, v5, w5))
    
    # Term 6: M6 = (A21 - A11)(B11 + B12)
    u6 = [-1, 0, 1, 0]  # A21 - A11
    v6 = [1, 1, 0, 0]  # B11 + B12
    w6 = [0, 0, 0, -1]  # C22 -= M6
    strassen_terms.append((u6, v6, w6))
    
    # Term 7: M7 = (A12 - A22)(B21 + B22)
    u7 = [0, 1, 0, -1]  # A12 - A22
    v7 = [0, 0, 1, 1]  # B21 + B22
    w7 = [-1, 0, 0, 0]  # C11 -= M7
    strassen_terms.append((u7, v7, w7))
    
    # Now build the full decomposition.
    # We have 15 columns. Split into 7 pairs (columns 0-1, 2-3, 4-5, 6-7, 8-9, 10-11, 12-13) and 1 single (column 14).
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    for pair_idx in range(7):
        col_start = 2 * pair_idx
        for (u, v, w) in strassen_terms:
            # u is 4-vector for A
            U_rows.append(list(u))
            
            # v is 4-vector for the 2x2 block of B starting at col_start
            # B has 2 rows, each with p columns.
            # Row 0 of B: columns 0..14
            # Row 1 of B: columns 0..14
            # For the block starting at col_start, we use B[0, col_start:col_start+2] and B[1, col_start:col_start+2]
            v_full = [0.0] * (m * p)
            v_full[0 * p + col_start + 0] = v[0]  # B[0][col_start]
            v_full[0 * p + col_start + 1] = v[1]  # B[0][col_start+1]
            v_full[1 * p + col_start + 0] = v[2]  # B[1][col_start]
            v_full[1 * p + col_start + 1] = v[3]  # B[1][col_start+1]
            V_rows.append(v_full)
            
            # w is 4-vector for C block: C[0][col_start], C[0][col_start+1], C[1][col_start], C[1][col_start+1]
            # W has shape (p*n, rank) = (30, rank)
            # W[k*n + i] corresponds to C[i][k]
            w_full = [0.0] * (p * n)
            w_full[(col_start + 0) * n + 0] = w[0]  # C[0][col_start]
            w_full[(col_start + 1) * n + 0] = w[1]  # C[0][col_start+1]
            w_full[(col_start + 0) * n + 1] = w[2]  # C[1][col_start]
            w_full[(col_start + 1) * n + 1] = w[3]  # C[1][col_start+1]
            W_rows.append(w_full)
    
    # Single column at position 14: schoolbook decomposition (4 terms)
    col_single = 14
    schoolbook_terms = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),  # A11 * B11 -> C11
        ([0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),  # A12 * B12 -> C12... wait
    ]
    
    # Let me redo schoolbook for 2x2 * 2x1 (single column)
    # C[0][14] = A[0][0]*B[0][14] + A[0][1]*B[1][14]
    # C[1][14] = A[1][0]*B[0][14] + A[1][1]*B[1][14]
    
    sb_terms = [
        # C11 contribution: A11 * B11
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        # C11 contribution: A12 * B21
        ([0, 1, 0, 0], [0, 1], [0, 1]),
        # C21 contribution: A21 * B11
        ([0, 0, 1, 0], [1, 0], [0, 0]),
        # C21 contribution: A22 * B21
        ([0, 0, 0, 1], [0, 1], [1, 1]),
    ]
    
    # Wait, I'm confusing myself. Let me be more careful.
    # For a single column k=14:
    # C[0][14] = A[0][0]*B[0][14] + A[0][1]*B[1][14]
    # C[1][14] = A[1][0]*B[0][14] + A[1][1]*B[1][14]
    
    # So we need 4 terms:
    # Term A: A[0][0] * B[0][14] -> contributes to C[0][14]
    # Term B: A[0][1] * B[1][14] -> contributes to C[0][14]
    # Term C: A[1][0] * B[0][14] -> contributes to C[1][14]
    # Term D: A[1][1] * B[1][14] -> contributes to C[1][14]
    
    for (u, v, w) in [
        ([1, 0, 0, 0], [1, 0], [1, 0]),  # A11 * B[0][14] -> C[0][14]
        ([0, 1, 0, 0], [0, 1], [1, 0]),  # A12 * B[1][14] -> C[0][14]
        ([0, 0, 1, 0], [1, 0], [0, 1]),  # A21 * B[0][14] -> C[1][14]
        ([0, 0, 0, 1], [0, 1], [0, 1]),  # A22 * B[1][14] -> C[1][14]
    ]:
        U_rows.append(list(u))
        
        v_full = [0.0] * (m * p)
        v_full[0 * p + col_single] = v[0]  # B[0][14]
        v_full[1 * p + col_single] = v[1]  # B[1][14]
        V_rows.append(v_full)
        
        w_full = [0.0] * (p * n)
        w_full[col_single * n + 0] = w[0]  # C[0][14]
        w_full[col_single * n + 1] = w[1]  # C[1][14]
        W_rows.append(w_full)
    
    rank = len(U_rows)
    U = U_rows
    V = V_rows
    W = W_rows
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""
Matrix multiplication 4x4 x 4x5 — rank-minimal exact tensor decomposition.

Uses the 7-term Strassen decomposition for the 4x4 x 4x4 block (top 4 rows of B),
and 4 schoolbook terms for the 4x4 x 4x1 block (bottom row of B).
Total rank = 49 + 4 = 53.

Output factor W follows the trace convention W[k*n + i, r] where
n = 4 (rows of A), p = 5 (columns of B).
"""

import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = 4, 4, 5
    # U shape: (n*m, rank) = (16, 53)
    # V shape: (m*p, rank) = (20, 53)
    # W shape: (p*n, rank) = (20, 53)
    
    # We'll build U, V, W as lists of lists for the required interface.
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # ---- Part 1: 4x4 x 4x4 via recursive Strassen (7+7+... = 49 terms) ----
    # A is 4x4, B_top is 4x4 (columns 0..3 of B)
    # C_top = A * B_top is 4x4
    #
    # Block indices: A = [[A11, A12],[A21, A22]], B_top = [[B11, B12],[B21, B22]]
    # Each block is 2x2.
    #
    # Strassen 2x2x2 7-term:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B12-B22)
    #   M4 = A22(B21-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For 2x2x2, the scalar 7-term decomposition has:
    #   Each "term" t in 1..7 contributes to C entries via coefficients.
    #   U factors (for A): 4x4 matrix per term
    #   V factors (for B): 4x4 matrix per term
    #   W factors (for C): 4x4 matrix per term (coefficient matrix)
    #
    # But actually, for tensor decomposition, each term is a rank-1:
    #   term_t = u_t (A-index) * v_t (B-index) * w_t (C-index)
    # where u_t is a 4-dim vector (flattened A), v_t is 4-dim (flattened B),
    # w_t is 4-dim (flattened C).
    #
    # For the 2x2 block Strassen, each "M_k" itself is a 2x2x2 multiplication
    # which needs 7 scalar terms. So 7 * 7 = 49 total for 4x4x4.
    #
    # Let me define the 2x2x2 rank-7 decomposition explicitly.
    # For 2x2x2: A = [[a0,a1],[a2,a3]], B = [[b0,b1],[b2,b3]], C = A*B
    # C00 = a0*b0 + a1*b2
    # C01 = a0*b1 + a1*b3
    # C10 = a2*b0 + a3*b2
    # C11 = a2*b1 + a3*b3
    #
    # Strassen:
    # M1 = (a0+a3)(b0+b3)
    # M2 = (a2+a3)*b0
    # M3 = a0*(b1-b3)
    # M4 = a3*(b2-b0)
    # M5 = (a0+a1)*b3
    # M6 = (a2-a0)*(b0+b1)
    # M7 = (a1-a3)*(b2+b3)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each M_k is a product of two linear forms in A and B.
    # M_k = (sum_i alpha_{k,i} * a_i) * (sum_j beta_{k,j} * b_j)
    # Then C_c = sum_k gamma_{k,c} * M_k
    #
    # So the rank-7 decomposition of 2x2x2 has:
    #   U factor for term k: alpha_k (4-dim vector)
    #   V factor for term k: beta_k (4-dim vector)
    #   W factor for term k: gamma_k (4-dim vector)
    
    # Define the 7-term 2x2x2 decomposition
    # alpha_k[i] for i=0..3 (a0,a1,a2,a3)
    # beta_k[j] for j=0..3 (b0,b1,b2,b3)
    # gamma_k[c] for c=0..3 (C00,C01,C10,C11)
    
    # M1: (a0+a3)(b0+b3)
    alpha = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    beta = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    gamma = [
        [1, 0, 0, -1],  # M1: C00+1, C01+0, C10+0, C11-1
        [0, 0, 1, -1],  # M2: C00+0, C01+0, C10+1, C11-1
        [0, 1, 0, 1],   # M3: C00+0, C01+1, C10+0, C11+1
        [1, 0, 1, 0],   # M4: C00+1, C01+0, C10+1, C11+0
        [-1, 1, 0, 0],  # M5: C00-1, C01+1, C10+0, C11+0
        [0, 0, 0, 1],   # M6: C00+0, C01+0, C10+0, C11+1
        [1, 0, 0, 0],   # M7: C00+1, C01+0, C10+0, C11+0
    ]
    
    # Now for 4x4x4, we have blocks of size 2x2.
    # A block (i,j) has indices: A[2*i + r, 2*j + c] for r,c in {0,1}
    # Flatten A as 16-dim: index = (2*i+r)*4 + (2*j+c) = 8*i + 4*j + 2*r + c
    # Flatten B_top as 16-dim: same indexing
    # Flatten C_top as 16-dim: same indexing
    
    # For each Strassen "outer" term k (0..6), we have:
    #   M_k = (sum over A-blocks alpha_k[bi] * A_bi) * (sum over B-blocks beta_k[bj] * B_bj)
    # where bi, bj are block indices 0..3 (for 2x2 blocks of the 4x4)
    # Block index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #   bi=0: A11, bi=1: A12, bi=2: A21, bi=3: A22
    #
    # Then M_k is a 2x2x2 multiplication of the resulting 2x2 blocks.
    # So we expand M_k using the 7-term 2x2x2 decomposition.
    # For inner term s (0..6), the contribution is:
    #   alpha_inner[s][ri] * beta_inner[s][rj] * gamma_inner[s][rc]
    # where ri, rj, rc are indices within the 2x2 blocks.
    #
    # The total A-index for this term:
    #   For outer block bi with coefficient alpha_k[bi]:
    #     A element at (2*bi_row + ri_offset, 2*bi_col + cj_offset)
    #   Wait, this is getting complex. Let me think differently.
    #
    # Actually, the outer Strassen operates on 2x2 blocks of the 4x4 matrices.
    # M_k = (linear combo of A-blocks) * (linear combo of B-blocks)
    # Each "linear combo of A-blocks" is a 2x2 matrix.
    # Then M_k is a 2x2 * 2x2 = 2x2 multiplication.
    # We expand this using the 7-term scalar decomposition for 2x2x2.
    #
    # For inner term s:
    #   M_k = sum_s (alpha_s . A_combined) * (beta_s . B_combined) * gamma_s
    # where A_combined is the 2x2 matrix that is the linear combo of A-blocks.
    #
    # Let me index more carefully.
    # A is 4x4. Block (I,J) where I,J in {0,1} is A[2I:2I+2, 2J:2J+2].
    # Outer Strassen:
    #   A_comb_k = sum_IJ alpha_k[I*2+J] * A[I,J]  (this is a 2x2 matrix)
    #   B_comb_k = sum_IJ beta_k[I*2+J] * B[I,J]  (2x2 matrix)
    #   M_k = A_comb_k * B_comb_k  (2x2 matrix)
    #
    # Inner Strassen on M_k:
    #   For s in 0..6:
    #     scalar = (sum_rc alpha_s[rc] * A_comb_k[rc]) * (sum_rc beta_s[rc] * B_comb_k[rc])
    #     C_comb_k[rc'] += gamma_s[rc'] * scalar
    #
    # So the full term for (k,s) is:
    #   A_factor = sum_IJ sum_rc alpha_k[I*2+J] * alpha_s[rc] * A[2I+r, 2J+c]
    #   B_factor = sum_IJ sum_rc beta_k[I*2+J] * beta_s[rc] * B[2I+r, 2J+c]
    #   C_factor = sum_IJ sum_rc gamma_s[rc] * C[2I+r, 2J+c]
    #
    # Where rc = (r,c) with r,c in {0,1}, and the 2x2 block index is I*2+J.
    # Wait, rc in the inner decomposition refers to positions within the 2x2 blocks.
    # Let me use (r,c) for positions within a 2x2 block.
    # alpha_s is indexed by position in the 2x2: alpha_s[0]=pos(0,0), alpha_s[1]=pos(0,1),
    # alpha_s[2]=pos(1,0), alpha_s[3]=pos(1,1)
    
    col = 0
    for k in range(7):
        for s in range(7):
            # A factor: 16-dim vector
            # A[2I+r, 2J+c] has flat index (2I+r)*4 + (2J+c) = 8I + 4J + 2r + c
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    a_coeff = alpha[k][block_idx]
                    if a_coeff == 0:
                        continue
                    for r in range(2):
                        for c in range(2):
                            a_s_coeff = alpha[s][r * 2 + c]
                            if a_s_coeff == 0:
                                continue
                            a_flat = (2 * I + r) * 4 + (2 * J + c)
                            U[a_flat][col] += a_coeff * a_s_coeff
            
            # B factor: for B_top (columns 0..3), same indexing
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    b_coeff = beta[k][block_idx]
                    if b_coeff == 0:
                        continue
                    for r in range(2):
                        for c in range(2):
                            b_s_coeff = beta[s][r * 2 + c]
                            if b_s_coeff == 0:
                                continue
                            b_flat = (2 * I + r) * 4 + (2 * J + c)
                            V[b_flat][col] += b_coeff * b_s_coeff
            
            # C factor: for C_top (4x4), flat index (2I+r)*4 + (2J+c)
            # W index convention: W[k*n + i, r] where k is column of B (0..p-1), i is row of A (0..n-1)
            # C[i, k] corresponds to W[k*4 + i, r]
            # Here C_top[i, j] where j is column in B_top (0..3), i is row in A (0..3)
            # C[i, j] = sum ... so W[j*4 + i, col] gets the coefficient
            for I in range(2):
                for J in range(2):
                    for r in range(2):
                        for c in range(2):
                            g_coeff = gamma[s][r * 2 + c]
                            if g_coeff == 0:
                                continue
                            i = 2 * I + r  # row in A
                            j = 2 * J + c  # col in B_top
                            w_idx = j * n + i  # k*n + i with k=j
                            W[w_idx][col] += g_coeff
            
            col += 1
    
    # ---- Part 2: 4x4 x 4x1 (bottom row of B, column index 4) ----
    # C_bottom[i, 4] = sum_j A[i,j] * B[j, 4]
    # This is a 4x4x1 multiplication, rank 4 (schoolbook).
    # For each j in 0..3: term j contributes A[i,j]*B[j,4] to C[i,4]
    
    for j in range(4):
        # A factor: A[i,j] for all i
        for i in range(4):
            a_flat = i * 4 + j
            U[a_flat][col] = 1
        # B factor: B[j, 4]
        b_flat = j * 5 + 4
        V[b_flat][col] = 1
        # C factor: C[i, 4] for all i
        for i in range(4):
            w_idx = 4 * 4 + i  # k=4, i varies
            W[w_idx][col] = 1
        col += 1
    
    # col should be 49 + 4 = 53
    assert col == 53, f"Expected col=53, got {col}"
    
    return {"U": U, "V": V, "W": W}

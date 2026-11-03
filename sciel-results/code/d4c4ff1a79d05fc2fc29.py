"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2_mult(U_rows, V_rows, W_rows, u_offset, v_offset, w_offset, rank_offset):
    """
    Apply Strassen's 7-multiplication algorithm for a 2x2 * 2x2 block.
    
    U_rows: list of row indices in U (length 4, representing 2x2)
    V_rows: list of row indices in V (length 4, representing 2x2)
    W_rows: list of row indices in W (length 4, representing 2x2)
    
    U is indexed as U[i*m + j] where i is row of A, j is col of A.
    For a 2x2 block at A-rows (r0, r1) and A-cols (c0, c1):
      U rows: [r0*m+c0, r0*m+c1, r1*m+c0, r1*m+c1]
    
    V is indexed as V[j*p + k] where j is col of A / row of B, k is col of B.
    For a 2x2 block at B-rows (c0, c1) and B-cols (d0, d1):
      V rows: [c0*p+d0, c0*p+d1, c1*p+d0, c1*p+d1]
    
    W is indexed as W[k*n + i] where k is col of B, i is row of A.
    For a 2x2 block at output-rows (d0, d1) and output-cols (r0, r1):
      W rows: [d0*n+r0, d0*n+r1, d1*n+r0, d1*n+r1]
    
    Returns list of (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w) tuples
    representing 7 rank-1 terms.
    """
    n, m, p = DIMENSIONS
    
    # U block indices: [u00, u01, u10, u11]
    u00, u01, u10, u11 = U_rows
    # V block indices: [v00, v01, v10, v11]
    v00, v01, v10, v11 = V_rows
    # W block indices: [w00, w01, w10, w11]
    w00, w01, w10, w11 = W_rows
    
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen product, we need to express the sum/difference of U and V entries
    # as linear combinations, and the W entry as a linear combination.
    
    # The tensor decomposition requires that for each rank-1 term:
    # U[:, c] * V[:, c] contributes to W[:, c] via outer product structure.
    # Specifically, W[k*n+i, c] = sum over terms where U[i*m+j, c]*V[j*p+k, c] contributes.
    
    # In Strassen, each "multiplication" M_s corresponds to a product of a linear form in A
    # and a linear form in B, contributing to a linear form in C.
    
    # We represent each Strassen term as a single rank-1 term in the decomposition.
    # The U vector for that term is the coefficients of the A linear form,
    # the V vector is the coefficients of the B linear form,
    # and the W vector is the coefficients of the C linear form.
    
    # Term 1: M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # U: A00+A11 -> U[u00]=1, U[u11]=1
    # V: B00+B11 -> V[v00]=1, V[v11]=1
    # W: C00 + C11 -> W[w00]=1, W[w11]=1
    
    # Term 2: M2 = (A10+A11)*B00 -> contributes to C10, C11 (with -1 for C11)
    # U: A10+A11 -> U[u10]=1, U[u11]=1
    # V: B00 -> V[v00]=1
    # W: C10 - C11 -> W[w10]=1, W[w11]=-1
    
    # Term 3: M3 = A00*(B01-B11) -> contributes to C01, C11
    # U: A00 -> U[u00]=1
    # V: B01-B11 -> V[v01]=1, V[v11]=-1
    # W: C01 + C11 -> W[w01]=1, W[w11]=1
    
    # Term 4: M4 = A11*(B10-B00) -> contributes to C00, C10
    # U: A11 -> U[u11]=1
    # V: B10-B00 -> V[v10]=1, V[v00]=-1
    # W: C00 + C10 -> W[w00]=1, W[w10]=1
    
    # Term 5: M5 = (A00+A01)*B11 -> contributes to C00 (with -1), C01
    # U: A00+A01 -> U[u00]=1, U[u01]=1
    # V: B11 -> V[v11]=1
    # W: -C00 + C01 -> W[w00]=-1, W[w01]=1
    
    # Term 6: M6 = (A10-A00)*(B00+B01) -> contributes to C11
    # U: A10-A00 -> U[u10]=1, U[u00]=-1
    # V: B00+B01 -> V[v00]=1, V[v01]=1
    # W: C11 -> W[w11]=1
    
    # Term 7: M7 = (A01-A11)*(B10+B11) -> contributes to C00
    # U: A01-A11 -> U[u01]=1, U[u11]=-1
    # V: B10+B11 -> V[v10]=1, V[v11]=1
    # W: C00 -> W[w00]=1
    
    terms = []
    
    # Term 1
    terms.append(([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)]))
    # Term 2
    terms.append(([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)]))
    # Term 3
    terms.append(([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)]))
    # Term 4
    terms.append(([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)]))
    # Term 5
    terms.append(([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)]))
    # Term 6
    terms.append(([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)]))
    # Term 7
    terms.append(([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)]))
    
    return terms


def construct():
    """
    Construct a rank-28 decomposition using Strassen's algorithm on 2x2 blocks.
    
    We partition the 2x4 * 4x4 multiplication into blocks:
    A is 2x4, B is 4x4, C is 2x4.
    
    We split A into two 2x2 blocks horizontally: A = [A_left, A_right]
    We split B into four 2x2 blocks: B = [[B_tl, B_tr], [B_bl, B_br]]
    We split C into two 2x2 blocks horizontally: C = [C_left, C_right]
    
    Then C_left = A_left * B_tl + A_right * B_bl
    C_right = A_left * B_tr + A_right * B_br
    
    Each product of a 2x2 and a 2x2 can be done with 7 multiplications (Strassen).
    But we have 4 such products, so naively 4*7 = 28.
    
    However, the products share structure. Let's think more carefully.
    
    Actually, let's just apply Strassen independently to each of the 4 2x2*2x2 multiplications:
    - A_left (2x2) * B_tl (2x2) -> contributes to C_left
    - A_right (2x2) * B_bl (2x2) -> contributes to C_left
    - A_left (2x2) * B_tr (2x2) -> contributes to C_right
    - A_right (2x2) * B_br (2x2) -> contributes to C_right
    
    Each uses 7 rank-1 terms, total 28.
    
    n=2, m=4, p=4.
    A has rows 0,1 and cols 0,1,2,3.
    B has rows 0,1,2,3 and cols 0,1,2,3.
    C has rows 0,1 and cols 0,1,2,3.
    
    A_left: rows 0,1; cols 0,1
    A_right: rows 0,1; cols 2,3
    B_tl: rows 0,1; cols 0,1
    B_bl: rows 2,3; cols 0,1
    B_tr: rows 0,1; cols 2,3
    B_br: rows 2,3; cols 2,3
    C_left: rows 0,1; cols 0,1
    C_right: rows 0,1; cols 2,3
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (16, rank)
    # W has shape (p*n, rank) = (8, rank)
    
    # We'll build lists of (row_index, coefficient) for each rank-1 term.
    U_terms = []  # list of lists of (row, coeff)
    V_terms = []
    W_terms = []
    
    # Block 1: A_left * B_tl -> C_left
    # A_left rows: 0,1; cols: 0,1
    # U indices: i*m+j for i in [0,1], j in [0,1]
    U_block1 = [0*4+0, 0*4+1, 1*4+0, 1*4+1]  # [0, 1, 4, 5]
    # B_tl rows: 0,1; cols: 0,1
    # V indices: j*p+k for j in [0,1], k in [0,1]
    V_block1 = [0*4+0, 0*4+1, 1*4+0, 1*4+1]  # [0, 1, 4, 5]
    # C_left rows: 0,1; cols: 0,1
    # W indices: k*n+i for k in [0,1], i in [0,1]
    W_block1 = [0*2+0, 0*2+1, 1*2+0, 1*2+1]  # [0, 1, 2, 3]
    
    for u, v, w in _strassen_2x2_mult(U_block1, V_block1, W_block1, 0, 0, 0, 0):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
    
    # Block 2: A_right * B_bl -> C_left
    # A_right rows: 0,1; cols: 2,3
    U_block2 = [0*4+2, 0*4+3, 1*4+2, 1*4+3]  # [2, 3, 6, 7]
    # B_bl rows: 2,3; cols: 0,1
    V_block2 = [2*4+0, 2*4+1, 3*4+0, 3*4+1]  # [8, 9, 12, 13]
    # C_left same as block 1
    W_block2 = W_block1  # [0, 1, 2, 3]
    
    for u, v, w in _strassen_2x2_mult(U_block2, V_block2, W_block2, 0, 0, 0, 0):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
    
    # Block 3: A_left * B_tr -> C_right
    # A_left same as block 1
    U_block3 = U_block1  # [0, 1, 4, 5]
    # B_tr rows: 0,1; cols: 2,3
    V_block3 = [0*4+2, 0*4+3, 1*4+2, 1*4+3]  # [2, 3, 6, 7]
    # C_right rows: 0,1; cols: 2,3
    W_block3 = [2*2+0, 2*2+1, 3*2+0, 3*2+1]  # [4, 5, 6, 7]
    
    for u, v, w in _strassen_2x2_mult(U_block3, V_block3, W_block3, 0, 0, 0, 0):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
    
    # Block 4: A_right * B_br -> C_right
    # A_right same as block 2
    U_block4 = U_block2  # [2, 3, 6, 7]
    # B_br rows: 2,3; cols: 2,3
    V_block4 = [2*4+2, 2*4+3, 3*4+2, 3*4+3]  # [10, 11, 14, 15]
    # C_right same as block 3
    W_block4 = W_block3  # [4, 5, 6, 7]
    
    for u, v, w in _strassen_2x2_mult(U_block4, V_block4, W_block4, 0, 0, 0, 0):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
    
    rank = len(U_terms)
    
    # Build the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (row, coeff) in U_terms[c]:
            U[row][c] = coeff
        for (row, coeff) in V_terms[c]:
            V[row][c] = coeff
        for (row, coeff) in W_terms[c]:
            W[row][c] = coeff
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

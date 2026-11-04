"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    The 3x3 * 3x6 multiplication can be viewed as two independent 3x3 * 3x3
    multiplications (splitting the 6 columns into two groups of 3).
    Each 3x3 * 3x3 multiplication has rank 7 via Strassen's algorithm.
    Total rank = 7 + 7 = 14.
    
    The multiplication tensor T has T[a,b,c] = 1 iff there exist i,j,k such that:
      a = i*m + j  (i in [0,n), j in [0,m))
      b = j*p + k  (j in [0,m), k in [0,p))
      c = k*n + i  (k in [0,p), i in [0,n))
    
    For n=3, m=3, p=6:
      a = 3*i + j
      b = 3*j + k  where j in {0,1,2}, k in {0,...,5}
      c = 6*k + i  where k in {0,...,5}, i in [0,1,2]
    
    We split k into k1 = k (for k in 0..2) and k2 = k-3 (for k in 3..5).
    Block 1: k in {0,1,2}, so b = 3*j + k, c = 6*k + i
    Block 2: k in {3,4,5}, so b = 3*j + (k-3), c = 6*k + i = 18 + 6*(k-3) + i
    
    For each block, we apply Strassen's algorithm for 3x3 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Strassen's algorithm for 3x3 matrix multiplication
    # C = A * B where A, B, C are 3x3
    # The 7 products are:
    # M1 = (A00 + A22) * (B00 + B22)
    # M2 = (A20 + A22) * B00
    # M3 = A00 * (B01 - B21)
    # M4 = A22 * (B11 - B01)
    # M5 = (A01 + A11) * B11
    # M6 = (A10 + A20) * (B00 + B10)
    # M7 = (A11 + A22) * (B11 + B21)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # C20 = M6 - M4 + M7
    # C21 = M2 + M6
    # C22 = M1 - M3 + M6
    
    # For 3x3 * 3x3, the tensor indices are:
    # a = i*3 + j (input A index)
    # b = j*3 + k (input B index)  
    # c = k*3 + i (output C index)
    
    # Strassen decomposition for one 3x3x3 block:
    # Each of the 7 terms has U, V, W vectors of length 9 (for 3x3 case)
    # We need to map these to the 3x3x6 problem.
    
    # For block 1 (k in 0..2):
    #   a = i*3 + j (same as 3x3 case)
    #   b = j*3 + k (same as 3x3 case, k in 0..2)
    #   c = 6*k + i (note: in 3x3 case it would be k*3 + i, here it's 6*k + i)
    
    # For block 2 (k in 3..5), let k' = k - 3 (k' in 0..2):
    #   a = i*3 + j (same)
    #   b = j*3 + (k'+3) = j*3 + k' + 3
    #   c = 6*(k'+3) + i = 18 + 6*k' + i
    
    # Let's define the Strassen terms for a general 3x3x3 block.
    # In the standard 3x3x3 tensor, indices are (a, b, c) where:
    #   a = i*3 + j, b = j*3 + k, c = k*3 + i
    #   with i,j,k in {0,1,2}
    
    # The 7 Strassen terms as (U, V, W) vectors in the 9-dimensional space:
    # Each term r has U_r[a], V_r[b], W_r[c]
    
    strassen_terms = []
    
    # Term 1: M1 = (A00 + A22) * (B00 + B22)
    # U: A00 (a=0) and A22 (a=8) get +1
    # V: B00 (b=0) and B22 (b=8) get +1
    # W: C00 (c=0), C11 (c=4), C22 (c=8) get +1
    #     C00 = M1 - M4 + M5 + M7, so M1 contributes +1 to C00
    #     C11 = M1 - M2 + M3 + M7, so M1 contributes +1 to C11
    #     C22 = M1 - M3 + M6, so M1 contributes +1 to C22
    U1 = [0]*9; U1[0] = 1; U1[8] = 1
    V1 = [0]*9; V1[0] = 1; V1[8] = 1
    W1 = [0]*9; W1[0] = 1; W1[4] = 1; W1[8] = 1
    strassen_terms.append((U1, V1, W1))
    
    # Term 2: M2 = (A20 + A22) * B00
    # U: A20 (a=6) and A22 (a=8) get +1
    # V: B00 (b=0) gets +1
    # W: C10 (c=3) and C21 (c=7) get +1
    #     C10 = M2 + M4, so M2 contributes +1 to C10
    #     C21 = M2 + M6, so M2 contributes +1 to C21
    U2 = [0]*9; U2[6] = 1; U2[8] = 1
    V2 = [0]*9; V2[0] = 1
    W2 = [0]*9; W2[3] = 1; W2[7] = 1
    strassen_terms.append((U2, V2, W2))
    
    # Term 3: M3 = A00 * (B01 - B21)
    # U: A00 (a=0) gets +1
    # V: B01 (b=1) gets +1, B21 (b=7) gets -1
    # W: C01 (c=1) and C11 (c=4) get +1, C22 (c=8) gets -1
    #     C01 = M3 + M5, so M3 contributes +1 to C01
    #     C11 = M1 - M2 + M3 + M7, so M3 contributes +1 to C11
    #     C22 = M1 - M3 + M6, so M3 contributes -1 to C22
    U3 = [0]*9; U3[0] = 1
    V3 = [0]*9; V3[1] = 1; V3[7] = -1
    W3 = [0]*9; W3[1] = 1; W3[4] = 1; W3[8] = -1
    strassen_terms.append((U3, V3, W3))
    
    # Term 4: M4 = A22 * (B11 - B01)
    # U: A22 (a=8) gets +1
    # V: B11 (b=4) gets +1, B01 (b=1) gets -1
    # W: C00 (c=0) gets -1, C10 (c=3) gets +1, C20 (c=6) gets -1
    #     C00 = M1 - M4 + M5 + M7, so M4 contributes -1 to C00
    #     C10 = M2 + M4, so M4 contributes +1 to C10
    #     C20 = M6 - M4 + M7, so M4 contributes -1 to C20
    U4 = [0]*9; U4[8] = 1
    V4 = [0]*9; V4[4] = 1; V4[1] = -1
    W4 = [0]*9; W4[0] = -1; W4[3] = 1; W4[6] = -1
    strassen_terms.append((U4, V4, W4))
    
    # Term 5: M5 = (A01 + A11) * B11
    # U: A01 (a=1) and A11 (a=4) get +1
    # V: B11 (b=4) gets +1
    # W: C00 (c=0) and C01 (c=1) get +1
    #     C00 = M1 - M4 + M5 + M7, so M5 contributes +1 to C00
    #     C01 = M3 + M5, so M5 contributes +1 to C01
    U5 = [0]*9; U5[1] = 1; U5[4] = 1
    V5 = [0]*9; V5[4] = 1
    W5 = [0]*9; W5[0] = 1; W5[1] = 1
    strassen_terms.append((U5, V5, W5))
    
    # Term 6: M6 = (A10 + A20) * (B00 + B10)
    # U: A10 (a=3) and A20 (a=6) get +1
    # V: B00 (b=0) and B10 (b=3) get +1
    # W: C20 (c=6) and C21 (c=7) get +1, C22 (c=8) gets +1
    #     C20 = M6 - M4 + M7, so M6 contributes +1 to C20
    #     C21 = M2 + M6, so M6 contributes +1 to C21
    #     C22 = M1 - M3 + M6, so M6 contributes +1 to C22
    U6 = [0]*9; U6[3] = 1; U6[6] = 1
    V6 = [0]*9; V6[0] = 1; V6[3] = 1
    W6 = [0]*9; W6[6] = 1; W6[7] = 1; W6[8] = 1
    strassen_terms.append((U6, V6, W6))
    
    # Term 7: M7 = (A11 + A22) * (B11 + B21)
    # U: A11 (a=4) and A22 (a=8) get +1
    # V: B11 (b=4) and B21 (b=7) get +1
    # W: C00 (c=0), C11 (c=4), C20 (c=6) get +1
    #     C00 = M1 - M4 + M5 + M7, so M7 contributes +1 to C00
    #     C11 = M1 - M2 + M3 + M7, so M7 contributes +1 to C11
    #     C20 = M6 - M4 + M7, so M7 contributes +1 to C20
    U7 = [0]*9; U7[4] = 1; U7[8] = 1
    V7 = [0]*9; V7[4] = 1; V7[7] = 1
    W7 = [0]*9; W7[0] = 1; W7[4] = 1; W7[6] = 1
    strassen_terms.append((U7, V7, W7))
    
    # Now build the full decomposition for 3x3 * 3x6.
    # U has shape (9, 14), V has shape (18, 14), W has shape (18, 14)
    
    rank = 14
    U = [[0.0] * rank for _ in range(n * m)]  # (9, 14)
    V = [[0.0] * rank for _ in range(m * p)]  # (18, 14)
    W = [[0.0] * rank for _ in range(p * n)]  # (18, 14)
    
    # For each block (block_idx = 0 or 1), and each Strassen term (term_idx = 0..6):
    # Block 0: k in 0..2
    #   U index: a = i*3 + j (unchanged from 3x3 case)
    #   V index: b = j*3 + k (unchanged from 3x3 case)
    #   W index: c = 6*k + i (note: in 3x3 case it's k*3 + i)
    #
    # Block 1: k in 3..5, k' = k-3
    #   U index: a = i*3 + j (unchanged)
    #   V index: b = j*3 + (k'+3) = j*3 + k' + 3
    #   W index: c = 6*(k'+3) + i = 18 + 6*k' + i
    
    for block_idx in range(2):
        for term_idx in range(7):
            U_t, V_t, W_t = strassen_terms[term_idx]
            col = block_idx * 7 + term_idx
            
            for a in range(9):
                U[a][col] = U_t[a]
            
            for b_idx in range(9):
                # b_idx corresponds to j*3 + k' in the 3x3 case
                j = b_idx // 3
                k_prime = b_idx % 3
                if block_idx == 0:
                    b_full = j * 3 + k_prime
                else:
                    b_full = j * 3 + k_prime + 3
                V[b_full][col] = V_t[b_idx]
            
            for c_idx in range(9):
                # c_idx corresponds to k'*3 + i in the 3x3 case
                k_prime = c_idx // 3
                i = c_idx % 3
                if block_idx == 0:
                    c_full = 6 * k_prime + i
                else:
                    c_full = 18 + 6 * k_prime + i
                W[c_full][col] = W_t[c_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

import numpy as np

def construct():
    n, m, p = 4, 4, 8
    
    # Rank for 4x4x4 using Strassen blocks is 49.
    # Total rank 98.
    rank = 98
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Define 2x2x2 Strassen factors
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Indices: i*2+j, j*2+k, k*2+i
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # Term 5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Block level Strassen for 4x4x4
    # M_c = A_part * B_part
    # A_part blocks, B_part blocks
    # c=0: A00+A11, B00+B11
    # c=1: A10+A11, B00
    # c=2: A00, B01-B11
    # c=3: A11, B10-B00
    # c=4: A00+A01, B11
    # c=5: A10-A00, B00+B01
    # c=6: A01-A11, B10+B11
    
    # W_block[c, block_k, block_i]
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    W_block = np.zeros((7, 4)) # 7 terms, 4 output blocks (00, 01, 10, 11)
    W_block[0, 0] = 1; W_block[0, 3] = 1
    W_block[1, 1] = 1; W_block[1, 3] = -1
    W_block[2, 2] = 1; W_block[2, 3] = 1
    W_block[3, 0] = 1; W_block[3, 1] = 1
    W_block[4, 0] = -1; W_block[4, 2] = 1
    W_block[5, 3] = 1
    W_block[6, 0] = 1
    
    # A_blocks[c, block_i, block_j]
    A_blocks = np.zeros((7, 4))
    A_blocks[0, 0] = 1; A_blocks[0, 3] = 1
    A_blocks[1, 2] = 1; A_blocks[1, 3] = 1
    A_blocks[2, 0] = 1
    A_blocks[3, 3] = 1
    A_blocks[4, 0] = 1; A_blocks[4, 1] = 1
    A_blocks[5, 2] = 1; A_blocks[5, 0] = -1
    A_blocks[6, 1] = 1; A_blocks[6, 3] = -1
    
    # B_blocks[c, block_j, block_k]
    B_blocks = np.zeros((7, 4))
    B_blocks[0, 0] = 1; B_blocks[0, 3] = 1
    B_blocks[1, 0] = 1
    B_blocks[2, 1] = 1; B_blocks[2, 3] = -1
    B_blocks[3, 2] = 1; B_blocks[3, 0] = -1
    B_blocks[4, 3] = 1
    B_blocks[5, 0] = 1; B_blocks[5, 1] = 1
    B_blocks[6, 2] = 1; B_blocks[6, 3] = 1
    
    # Construct U, V, W for 4x4x4 (rank 49)
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for c in range(7):
        for r in range(7):
            col = c * 7 + r
            # U
            for bi in range(2):
                for bj in range(2):
                    if A_blocks[c, bi*2+bj] != 0:
                        # Local indices i, j
                        for li in range(2):
                            for lj in range(2):
                                U44[(bi*2+li)*2 + (bj*2+lj), col] = U2[li*2+lj, r] * A_blocks[c, bi*2+bj]
            # V
            for bj in range(2):
                for bk in range(2):
                    if B_blocks[c, bj*2+bk] != 0:
                        for lj in range(2):
                            for lk in range(2):
                                V44[(bj*2+lj)*2 + (bk*2+lk), col] = V2[lj*2+lk, r] * B_blocks[c, bj*2+bk]
            # W
            for bk in range(2):
                for bi in range(2):
                    # Output block (bi, bk)
                    # W_block index: c, block_idx
                    # Block index: bi*2 + bk
                    coeff = W_block[c, bi*2+bk]
                    if coeff != 0:
                        for lk in range(2):
                            for li in range(2):
                                # W2 index: k*2 + i
                                W44[(bk*2+lk)*2 + (bi*2+li), col] = W2[lk*2+li, r] * coeff

    # Now expand to 4x4x8
    # U is 16x98
    # V is 32x98
    # W is 32x98
    
    U[:, 0:49] = U44
    U[:, 49:98] = U44
    
    # V: first 16 rows for B1, next 16 for B2
    V[0:16, 0:49] = V44
    V[16:32, 49:98] = V44
    
    # W: first 16 rows for C1 (k=0..3), next 16 for C2 (k=4..7)
    # W44 indices are k*4+i. k in 0..3.
    # For C1, k_global = k_local.
    # For C2, k_global = k_local + 4.
    W[0:16, 0:49] = W44
    W[16:32, 49:98] = W44
    
    return {"U": U, "V": V, "W": W}

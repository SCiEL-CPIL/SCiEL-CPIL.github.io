import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # We estimate rank 88.
    # Let's allocate slightly more to be safe, or calculate exact.
    # Row 0: 1x4x4 (16) + 1x4x4 (16) = 32
    # Row 1-2: 2x4x4 (28) + 2x4x4 (28) = 56
    # Total 88.
    
    rank = 88
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen 2x2x2 vectors
    # U: (i,j) in 0..1 x 0..1 -> idx i*2+j
    # V: (j,k) in 0..1 x 0..1 -> idx j*2+k
    # W: (k,i) in 0..1 x 0..1 -> idx k*2+i
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U vectors
    u_s = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # V vectors
    v_s = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 1, -1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    # W vectors
    w_s = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, -1],# M2: C10, -C11
        [0, 1, 0, 1], # M3: C01, C11
        [1, 0, 1, 0], # M4: C00, C10
        [-1, 1, 0, 0],# M5: -C00, C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    def add_strassen_block(i_start, j_start, k_start, col):
        # i_start, j_start, k_start are global starts
        # 2x2x2 block
        for r in range(7):
            # U
            for li in range(2):
                for lj in range(2):
                    val = u_s[r][li*2+lj]
                    if val != 0:
                        gi = i_start + li
                        gj = j_start + lj
                        U[gi*m + gj, col] += val
            # V
            for lj in range(2):
                for lk in range(2):
                    val = v_s[r][lj*2+lk]
                    if val != 0:
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj*p + gk, col] += val
            # W
            for lk in range(2):
                for li in range(2):
                    val = w_s[r][lk*2+li]
                    if val != 0:
                        gk = k_start + lk
                        gi = i_start + li
                        W[gk*n + gi, col] += val
            col += 1
        return col

    def add_naive_1x4x4(i_start, j_start, k_start, col):
        # i is fixed (1 row)
        # j is 0..3
        # k is 0..3 (local)
        # C_k = sum_j A_j B_jk
        # Rank 16
        for lj in range(4):
            for lk in range(4):
                # U: A_i, j
                U[i_start*m + (j_start+lj), col] = 1
                # V: B_j, k
                V[(j_start+lj)*p + (k_start+lk), col] = 1
                # W: C_k, i
                W[(k_start+lk)*n + i_start, col] = 1
                col += 1
        return col

    # Block 1: Row 0, Cols 0-3 (Inner 0-3)
    # 1x4x4
    col = add_naive_1x4x4(0, 0, 0, col)
    
    # Block 2: Row 0, Cols 4-7 (Inner 0-3, Outer 4-7)
    # 1x4x4
    col = add_naive_1x4x4(0, 0, 4, col)
    
    # Block 3: Rows 1-2, Cols 0-3
    # 2x4x4
    # Split Inner 4 into 2+2. Split Outer 4 into 2+2.
    # 4 blocks of 2x2x2.
    # Inner 0-1, Outer 0-1
    col = add_strassen_block(1, 0, 0, col)
    # Inner 0-1, Outer 2-3
    col = add_strassen_block(1, 0, 2, col)
    # Inner 2-3, Outer 0-1
    col = add_strassen_block(1, 2, 0, col)
    # Inner 2-3, Outer 2-3
    col = add_strassen_block(1, 2, 2, col)
    
    # Block 4: Rows 1-2, Cols 4-7
    # 2x4x4
    # Inner 0-1, Outer 4-5
    col = add_strassen_block(1, 0, 4, col)
    # Inner 0-1, Outer 6-7
    col = add_strassen_block(1, 0, 6, col)
    # Inner 2-3, Outer 4-5
    col = add_strassen_block(1, 2, 4, col)
    # Inner 2-3, Outer 6-7
    col = add_strassen_block(1, 2, 6, col)
    
    return {'U': U, 'V': V, 'W': W}

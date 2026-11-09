import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 4 blocks of 2x4x2 using Strassen (2x2x2 x 2) -> 4 * (2*7) = 56? No.
    # Each 2x4x2 is split into two 2x2x2.
    # Block 1: i in 0,1; k in 0,1. j in 0,1 and j in 2,3.
    #   j in 0,1: 2x2x2 Strassen (rank 7)
    #   j in 2,3: 2x2x2 Strassen (rank 7)
    #   Total for i<2, k<2: 14
    # Block 2: i in 2,3; k in 2,3. j in 0,1 and j in 2,3.
    #   j in 0,1: 2x2x2 Strassen (rank 7)
    #   j in 2,3: 2x2x2 Strassen (rank 7)
    #   Total for i>=2, k>=2: 14
    # Block 3: i in 0,1; k in 4. j in 0..3.
    #   2x4x1 Schoolbook (rank 8)
    # Block 4: i in 2,3; k in 4. j in 0..3.
    #   2x4x1 Schoolbook (rank 8)
    # Total Rank = 14 + 14 + 8 + 8 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a 2x2x2 Strassen block
    # i_offset: offset for i index in A and C
    # j_offset: offset for j index in A and B
    # k_offset: offset for k index in B and C
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal current_rank
        # Local indices 0,1
        # Strassen factors
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        # M3: A00(B10-B11) -> C01 += M3, C11 += M3
        # M4: A11(B01+B11) -> C00 += M4, C10 += M4
        # M5: (A00+A01)B11 -> C00 -= M5, C01 += M5
        # M6: (A10-A00)(B00+B01) -> C11 += M6
        # M7: (A01-A11)(B10+B11) -> C00 -= M7
        
        factors = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: (i,j) pairs, V: (j,k) pairs, W: (k,i) pairs
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):-1})
        ]
        
        for u_coeffs, v_coeffs, w_coeffs in factors:
            for (li, lj), val_u in u_coeffs.items():
                gi = li + i_off
                gj = lj + j_off
                U[gi * m + gj][current_rank] += val_u
            for (lj, lk), val_v in v_coeffs.items():
                gj = lj + j_off
                gk = lk + k_off
                V[gj * p + gk][current_rank] += val_v
            for (lk, li), val_w in w_coeffs.items():
                gk = lk + k_off
                gi = li + i_off
                W[gk * n + gi][current_rank] += val_w
            current_rank += 1

    # Helper to add a 2x4x1 Schoolbook block
    # i_offset: offset for i index in A and C
    # j_offset: offset for j index in A and B (0)
    # k_val: k index in B and C
    def add_schoolbook_2x4x1(i_off, j_off, k_val):
        nonlocal current_rank
        for li in range(2):
            for lj in range(4):
                gi = li + i_off
                gj = lj + j_off
                gk = k_val
                U[gi * m + gj][current_rank] = 1
                V[gj * p + gk][current_rank] = 1
                W[gk * n + gi][current_rank] = 1
                current_rank += 1

    # Block 1: i in 0,1; k in 0,1
    # Split j into 0,1 and 2,3
    add_strassen_2x2x2(0, 0, 0)
    add_strassen_2x2x2(0, 2, 0)
    
    # Block 2: i in 2,3; k in 2,3
    # Split j into 0,1 and 2,3
    add_strassen_2x2x2(2, 0, 2)
    add_strassen_2x2x2(2, 2, 2)
    
    # Block 3: i in 0,1; k in 4
    add_schoolbook_2x4x1(0, 0, 4)
    
    # Block 4: i in 2,3; k in 4
    add_schoolbook_2x4x1(2, 0, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

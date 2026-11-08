import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = 2, 3, 4
    rank = 22
    
    # Index mapping
    # A[i,j] -> row i*m + j in U  (i=0..1, j=0..2)
    # B[j,k] -> row j*p + k in V  (j=0..2, k=0..3)
    # C[i,k] -> row k*n + i in W  (k=0..3, i=0..1)
    
    def a_idx(i, j): return i * m + j
    def b_idx(j, k): return j * p + k
    def c_idx(i, k): return k * n + i
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_list, v_list, w_list, col):
        for idx, val in u_list:
            U[idx][col] += val
        for idx, val in v_list:
            V[idx][col] += val
        for idx, val in w_list:
            W[idx][col] += val
            
    # Decomposition structure:
    # 1. Block (0,0) x (0,0) -> C(0,0) part: A[0:2, 0:2] * B[0:2, 0:2]
    #    Indices: A00,A01,A10,A11 and B00,B01,B10,B11
    #    Target: C00, C01, C10, C11
    #    Using Strassen-like (rank 7)
    
    # 2. Block (0,0) x (0,2) -> C(0,2) part: A[0:2, 0:2] * B[0:2, 2:4]
    #    Indices: A00,A01,A10,A11 and B02,B03,B12,B13
    #    Target: C02, C03, C12, C13
    #    Using Strassen-like (rank 7)
    
    # 3. Block (0,2) x (2,0) -> C(0,0) tail: A[0:2, 2:3] * B[2:3, 0:4]
    #    Indices: A02,A12 and B20,B21,B22,B23
    #    Target: C00, C01, C02, C03 (row 0) and C10, C11, C12, C13 (row 1)
    #    Using Outer Product (rank 8)
    
    # Total rank: 7 + 7 + 8 = 22
    
    # --- Part 1: A[0:2, 0:2] * B[0:2, 0:2] ---
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=4, B11=5
    # C00=0, C01=2, C10=1, C11=3
    
    # M1: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    #     Contributes to C00 (A00B00) and C11 (A11B11) and cross terms
    #     In Strassen, we combine to get all 4.
    #     Let's use the standard Strassen formulas mapped to our indices.
    #     C00 = M1 - M4 + M7
    #     C01 = M3 + M5
    #     C10 = M2 + M4
    #     C11 = M1 - M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    add_term([(a_idx(0,0), 1), (a_idx(1,1), 1)], [(b_idx(0,0), 1), (b_idx(1,1), 1)], [], 0)
    # M2 = (A10+A11)B00
    add_term([(a_idx(1,0), 1), (a_idx(1,1), 1)], [(b_idx(0,0), 1)], [], 1)
    # M3 = A00(B01-B11)
    add_term([(a_idx(0,0), 1)], [(b_idx(0,1), 1), (b_idx(1,1), -1)], [], 2)
    # M4 = A11(B10-B00)
    add_term([(a_idx(1,1), 1)], [(b_idx(1,0), 1), (b_idx(0,0), -1)], [], 3)
    # M5 = (A00+A01)B11
    add_term([(a_idx(0,0), 1), (a_idx(0,1), 1)], [(b_idx(1,1), 1)], [], 4)
    # M6 = (A10-A00)(B00+B01)
    add_term([(a_idx(1,0), 1), (a_idx(0,0), -1)], [(b_idx(0,0), 1), (b_idx(0,1), 1)], [], 5)
    # M7 = (A01-A11)(B10+B11)
    add_term([(a_idx(0,1), 1), (a_idx(1,1), -1)], [(b_idx(1,0), 1), (b_idx(1,1), 1)], [], 6)
    
    # Output W for Part 1:
    # C00 (idx 0): M1 - M4 + M7
    add_term([], [], [(c_idx(0,0), 1), (c_idx(1,0), -1), (c_idx(0,1), 1)], 0)
    add_term([], [], [(c_idx(0,0), -1), (c_idx(1,0), 1)], 3)
    add_term([], [], [(c_idx(0,0), 1)], 6)
    
    # C01 (idx 2): M3 + M5
    add_term([], [], [(c_idx(0,1), 1)], 2)
    add_term([], [], [(c_idx(0,1), 1)], 4)
    
    # C10 (idx 1): M2 + M4
    add_term([], [], [(c_idx(1,0), 1)], 1)
    add_term([], [], [(c_idx(1,0), 1)], 3)
    
    # C11 (idx 3): M1 - M3 + M6
    add_term([], [], [(c_idx(1,1), 1)], 0)
    add_term([], [], [(c_idx(1,1), -1)], 2)
    add_term([], [], [(c_idx(1,1), 1)], 5)

    # --- Part 2: A[0:2, 0:2] * B[0:2, 2:4] ---
    # A00=0, A01=1, A10=3, A11=4
    # B02=2, B03=3, B12=6, B13=7
    # C02=4, C03=6, C12=5, C13=7
    
    # Same Strassen structure, shifted B and C indices
    # M1' = (A00+A11)(B02+B13)
    add_term([(a_idx(0,0), 1), (a_idx(1,1), 1)], [(b_idx(0,2), 1), (b_idx(1,3), 1)], [], 7)
    # M2' = (A10+A11)B02
    add_term([(a_idx(1,0), 1), (a_idx(1,1), 1)], [(b_idx(0,2), 1)], [], 8)
    # M3' = A00(B03-B13)
    add_term([(a_idx(0,0), 1)], [(b_idx(0,3), 1), (b_idx(1,3), -1)], [], 9)
    # M4' = A11(B12-B02)
    add_term([(a_idx(1,1), 1)], [(b_idx(1,2), 1), (b_idx(0,2), -1)], [], 10)
    # M5' = (A00+A01)B13
    add_term([(a_idx(0,0), 1), (a_idx(0,1), 1)], [(b_idx(1,3), 1)], [], 11)
    # M6' = (A10-A00)(B02+B03)
    add_term([(a_idx(1,0), 1), (a_idx(0,0), -1)], [(b_idx(0,2), 1), (b_idx(0,3), 1)], [], 12)
    # M7' = (A01-A11)(B12+B13)
    add_term([(a_idx(0,1), 1), (a_idx(1,1), -1)], [(b_idx(1,2), 1), (b_idx(1,3), 1)], [], 13)
    
    # Output W for Part 2:
    # C02 (idx 4): M1' - M4' + M7'
    add_term([], [], [(c_idx(0,2), 1), (c_idx(1,2), -1), (c_idx(0,3), 1)], 7)
    add_term([], [], [(c_idx(0,2), -1), (c_idx(1,2), 1)], 10)
    add_term([], [], [(c_idx(0,2), 1)], 13)
    
    # C03 (idx 6): M3' + M5'
    add_term([], [], [(c_idx(0,3), 1)], 9)
    add_term([], [], [(c_idx(0,3), 1)], 11)
    
    # C12 (idx 5): M2' + M4'
    add_term([], [], [(c_idx(1,2), 1)], 8)
    add_term([], [], [(c_idx(1,2), 1)], 10)
    
    # C13 (idx 7): M1' - M3' + M6'
    add_term([], [], [(c_idx(1,3), 1)], 7)
    add_term([], [], [(c_idx(1,3), -1)], 9)
    add_term([], [], [(c_idx(1,3), 1)], 12)

    # --- Part 3: A[0:2, 2:3] * B[2:3, 0:4] ---
    # A02=2, A12=5
    # B20=8, B21=9, B22=10, B23=11
    # C00=0, C01=2, C02=4, C03=6 (from A02)
    # C10=1, C11=3, C12=5, C13=7 (from A12)
    
    # Outer product terms: A02 * B2k -> C0k
    # A12 * B2k -> C1k
    
    # Col 14: A02 * B20 -> C00
    add_term([(a_idx(0,2), 1)], [(b_idx(2,0), 1)], [(c_idx(0,0), 1)], 14)
    # Col 15: A02 * B21 -> C01
    add_term([(a_idx(0,2), 1)], [(b_idx(2,1), 1)], [(c_idx(0,1), 1)], 15)
    # Col 16: A02 * B22 -> C02
    add_term([(a_idx(0,2), 1)], [(b_idx(2,2), 1)], [(c_idx(0,2), 1)], 16)
    # Col 17: A02 * B23 -> C03
    add_term([(a_idx(0,2), 1)], [(b_idx(2,3), 1)], [(c_idx(0,3), 1)], 17)
    
    # Col 18: A12 * B20 -> C10
    add_term([(a_idx(1,2), 1)], [(b_idx(2,0), 1)], [(c_idx(1,0), 1)], 18)
    # Col 19: A12 * B21 -> C11
    add_term([(a_idx(1,2), 1)], [(b_idx(2,1), 1)], [(c_idx(1,1), 1)], 19)
    # Col 20: A12 * B22 -> C12
    add_term([(a_idx(1,2), 1)], [(b_idx(2,2), 1)], [(c_idx(1,2), 1)], 20)
    # Col 21: A12 * B23 -> C13
    add_term([(a_idx(1,2), 1)], [(b_idx(2,3), 1)], [(c_idx(1,3), 1)], 21)
    
    return {"U": U, "V": V, "W": W}

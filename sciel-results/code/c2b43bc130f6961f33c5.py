import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Rank 42 decomposition
    # 25 terms for 3x3 core (using Strassen for 2x2 blocks)
    # 17 terms for extra 2 columns (using Strassen for 2x2 blocks of A and B_extra)
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 42
    V = [[0] * rank for _ in range(m * p)]  # 15 x 42
    W = [[0] * rank for _ in range(p * n)]  # 15 x 42
    
    # Helper to add a term
    # u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w, term_idx
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs, term_idx):
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][term_idx] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][term_idx] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][term_idx] += coeff

    # --- Part 1: 3x3 Core (Columns 0-2 of B) ---
    # A = [[A11, A12], [A21, A22]]
    # A11: [[A00, A01], [A10, A11]] (indices 0,1,3,4)
    # A12: [[A02], [A12]] (indices 2,5)
    # A21: [[A20, A21]] (indices 6,7)
    # A22: [[A22]] (index 8)
    
    # B_core = [[B11, B12], [B21, B22]]
    # B11: [[B00, B01], [B10, B11]] (indices 0,1,5,6)
    # B12: [[B02], [B12]] (indices 2,7)
    # B21: [[B20, B21]] (indices 10,11)
    # B22: [[B22]] (index 12)
    
    # C_core = [[C11, C12], [C21, C22]]
    # C11: [[C00, C01], [C10, C11]] (indices 0,3,1,4) -> W[k*n+i]
    # C12: [[C02], [C12]] (indices 2,5)
    # C21: [[C20, C21]] (indices 6,9)
    # C22: [[C22]] (index 12)
    
    # Strassen terms for A11 * B11
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Term 0: M1
    add_term([0, 4], [1, 1], [0, 6], [1, 1], [0, 3, 1, 4], [1, 1, 1, 1], 0)
    
    # Term 1: M2
    add_term([3, 4], [1, 1], [0], [1], [1, 4], [1, 1], 1)
    
    # Term 2: M3
    add_term([0], [1], [1, 6], [1, -1], [2, 5], [1, 1], 2)
    
    # Term 3: M4
    add_term([4], [1], [5, 0], [1, -1], [0, 3, 1, 4], [1, 1, 1, 1], 3)
    
    # Term 4: M5
    add_term([0, 1], [1, 1], [6], [1], [0, 3, 1, 4], [-1, -1, -1, -1], 4)
    
    # Term 5: M6
    add_term([3, 0], [1, -1], [0, 1], [1, 1], [4], [1], 5)
    
    # Term 6: M7
    add_term([1, 4], [1, -1], [5, 6], [1, 1], [0, 3, 1, 4], [1, 1, 1, 1], 6)
    
    # Cross terms for 3x3 core
    # C11 += A12 * B21 (Outer product 2x1 * 1x2)
    # A12: [A02, A12] (2,5)
    # B21: [B20, B21] (10,11)
    # C11: C00(0), C01(3), C10(1), C11(4)
    # Term 7: A02 * B20 -> C00
    add_term([2], [1], [10], [1], [0], [1], 7)
    # Term 8: A02 * B21 -> C01
    add_term([2], [1], [11], [1], [3], [1], 8)
    # Term 9: A12 * B20 -> C10
    add_term([5], [1], [10], [1], [1], [1], 9)
    # Term 10: A12 * B21 -> C11
    add_term([5], [1], [11], [1], [4], [1], 10)
    
    # C12 += A12 * B22
    # A12: [A02, A12] (2,5)
    # B22: B22 (12)
    # C12: C02(2), C12(5)
    # Term 11: A02 * B22 -> C02
    add_term([2], [1], [12], [1], [2], [1], 11)
    # Term 12: A12 * B22 -> C12
    add_term([5], [1], [12], [1], [5], [1], 12)
    
    # C21 += A21 * B11
    # A21: [A20, A21] (6,7)
    # B11: [[B00, B01], [B10, B11]] (0,1,5,6)
    # C21: C20(6), C21(9)
    # Term 13: A20 * B00 -> C20
    add_term([6], [1], [0], [1], [6], [1], 13)
    # Term 14: A20 * B01 -> C21
    add_term([6], [1], [1], [1], [9], [1], 14)
    # Term 15: A20 * B10 -> C20
    add_term([6], [1], [5], [1], [6], [1], 15)
    # Term 16: A20 * B11 -> C21
    add_term([6], [1], [6], [1], [9], [1], 16)
    # Term 17: A21 * B00 -> C20
    add_term([7], [1], [0], [1], [6], [1], 17)
    # Term 18: A21 * B01 -> C21
    add_term([7], [1], [1], [1], [9], [1], 18)
    # Term 19: A21 * B10 -> C20
    add_term([7], [1], [5], [1], [6], [1], 19)
    # Term 20: A21 * B11 -> C21
    add_term([7], [1], [6], [1], [9], [1], 20)
    
    # C21 += A22 * B21
    # A22: A22 (8)
    # B21: [B20, B21] (10,11)
    # C21: C20(6), C21(9)
    # Term 21: A22 * B20 -> C20
    add_term([8], [1], [10], [1], [6], [1], 21)
    # Term 22: A22 * B21 -> C21
    add_term([8], [1], [11], [1], [9], [1], 22)
    
    # C22 += A21 * B12
    # A21: [A20, A21] (6,7)
    # B12: [B02, B12] (2,7)
    # C22: C22(12)
    # Term 23: A20 * B02 -> C22
    add_term([6], [1], [2], [1], [12], [1], 23)
    # Term 24: A20 * B12 -> C22
    add_term([6], [1], [7], [1], [12], [1], 24)
    # Term 25: A21 * B02 -> C22
    add_term([7], [1], [2], [1], [12], [1], 25)
    # Term 26: A21 * B12 -> C22
    add_term([7], [1], [7], [1], [12], [1], 26)
    
    # C22 += A22 * B22
    # Term 27: A22 * B22 -> C22
    add_term([8], [1], [12], [1], [12], [1], 27)
    
    # --- Part 2: Extra Columns (3-4 of B) ---
    # B_extra = [[B_extra_11, B_extra_12], [B_extra_21, B_extra_22]]
    # B_extra_11: [[B03, B04], [B13, B14]] (indices 3,4,8,9)
    # B_extra_12: None (since B is 3x5, B_extra is 3x2)
    # Wait, B is 3x5.
    # B = [B_core (3x3) | B_extra (3x2)]
    # B_extra rows:
    # Row 0: B03, B04 (indices 3,4)
    # Row 1: B13, B14 (indices 8,9)
    # Row 2: B23, B24 (indices 13,14)
    
    # C_extra = A * B_extra
    # C_extra is 3x2.
    # C_extra[0,0] = C03 (W index 3*3+0 = 9)
    # C_extra[0,1] = C04 (W index 4*3+0 = 12)
    # C_extra[1,0] = C13 (W index 3*3+1 = 10)
    # C_extra[1,1] = C14 (W index 4*3+1 = 13)
    # C_extra[2,0] = C23 (W index 3*3+2 = 11)
    # C_extra[2,1] = C24 (W index 4*3+2 = 14)
    
    # We use Strassen for A11 * B_extra_11 (2x2 * 2x2)
    # A11: [[A00, A01], [A10, A11]] (0,1,3,4)
    # B_extra_11: [[B03, B04], [B13, B14]] (3,4,8,9)
    
    # M1' = (A00+A11)(B03+B13)
    # M2' = (A10+A11)B03
    # M3' = A00(B04-B14)
    # M4' = A11(B13-B03)
    # M5' = (A00+A01)B14
    # M6' = (A10-A00)(B03+B04)
    # M7' = (A01-A11)(B13+B14)
    
    # C_extra_11 (2x2 part of C_extra)
    # C_extra_11[0,0] = C03 (9)
    # C_extra_11[0,1] = C04 (12)
    # C_extra_11[1,0] = C13 (10)
    # C_extra_11[1,1] = C14 (13)
    
    # C_extra_11 = M1' + M4' - M5' + M7' (for [0,0])
    # C_extra_11[0,0] = M1' + M4' - M5' + M7'
    # C_extra_11[0,1] = M3' + M5'
    # C_extra_11[1,0] = M2' + M4'
    # C_extra_11[1,1] = M1' - M2' + M3' + M6'
    
    # Term 28: M1'
    add_term([0, 4], [1, 1], [3, 8], [1, 1], [9, 10, 12, 13], [1, 1, 1, 1], 28)
    
    # Term 29: M2'
    add_term([3, 4], [1, 1], [3], [1], [10, 13], [1, 1], 29)
    
    # Term 30: M3'
    add_term([0], [1], [4, 9], [1, -1], [12, 13], [1, 1], 30)
    
    # Term 31: M4'
    add_term([4], [1], [8, 3], [1, -1], [9, 10, 12, 13], [1, 1, 1, 1], 31)
    
    # Term 32: M5'
    add_term([0, 1], [1, 1], [9], [1], [9, 10, 12, 13], [-1, -1, -1, -1], 32)
    
    # Term 33: M6'
    add_term([3, 0], [1, -1], [3, 4], [1, 1], [13], [1], 33)
    
    # Term 34: M7'
    add_term([1, 4], [1, -1], [8, 9], [1, 1], [9, 10, 12, 13], [1, 1, 1, 1], 34)
    
    # Cross terms for Extra
    # C_extra[0,0] (C03) += A12 * B_extra_21
    # A12: [A02, A12] (2,5)
    # B_extra_21: B23 (13)
    # C03: W[9]
    # Term 35: A02 * B23 -> C03
    add_term([2], [1], [13], [1], [9], [1], 35)
    # Term 36: A12 * B23 -> C13 (10)
    add_term([5], [1], [13], [1], [10], [1], 36)
    
    # C_extra[0,1] (C04) += A12 * B_extra_22
    # B_extra_22: B24 (14)
    # C04: W[12]
    # Term 37: A02 * B24 -> C04
    add_term([2], [1], [14], [1], [12], [1], 37)
    # Term 38: A12 * B24 -> C14 (13)
    add_term([5], [1], [14], [1], [13], [1], 38)
    
    # C_extra[1,0] (C13) += A21 * B_extra_11
    # A21: [A20, A21] (6,7)
    # B_extra_11: [[B03, B04], [B13, B14]] (3,4,8,9)
    # C13: W[10]
    # Term 39: A20 * B03 -> C13
    add_term([6], [1], [3], [1], [10], [1], 39)
    # Term 40: A20 * B04 -> C14 (13)
    add_term([6], [1], [4], [1], [13], [1], 40)
    # Term 41: A20 * B13 -> C13
    add_term([6], [1], [8], [1], [10], [1], 41)
    # Term 42: A20 * B14 -> C14
    add_term([6], [1], [9], [1], [13], [1], 42)
    # Term 43: A21 * B03 -> C13
    add_term([7], [1], [3], [1], [10], [1], 43)
    # Term 44: A21 * B04 -> C14
    add_term([7], [1], [4], [1], [13], [1], 44)
    # Term 45: A21 * B13 -> C13
    add_term([7], [1], [8], [1], [10], [1], 45)
    # Term 46: A21 * B14 -> C14
    add_term([7], [1], [9], [1], [13], [1], 46)
    
    # C_extra[1,0] (C13) += A22 * B_extra_21
    # A22: A22 (8)
    # B_extra_21: B23 (13)
    # Term 47: A22 * B23 -> C13
    add_term([8], [1], [13], [1], [10], [1], 47)
    
    # C_extra[1,1] (C14) += A22 * B_extra_22
    # B_extra_22: B24 (14)
    # Term 48: A22 * B24 -> C14
    add_term([8], [1], [14], [1], [13], [1], 48)
    
    # C_extra[2,0] (C23) += A21 * B_extra_12? No, B_extra_12 doesn't exist.
    # C_extra[2,0] (C23) is part of C_extra_21?
    # C_extra = [C_extra_11 (2x2) | C_extra_12 (2x1)]
    #           [C_extra_21 (1x2) | C_extra_22 (1x1)]
    
    # I missed C_extra_21 and C_extra_22.
    # C_extra_21: C23 (11), C24 (14)
    # C_extra_22: None (1x1)
    
    # C_extra_21 = A21 * B_extra_11 + A22 * B_extra_21
    # A21: [A20, A21] (6,7)
    # B_extra_11: [[B03, B04], [B13, B14]] (3,4,8,9)
    # C23: W[11]
    # C24: W[14]
    
    # Term 49: A20 * B03 -> C23
    add_term([6], [1], [3], [1], [11], [1], 49)
    # Term 50: A20 * B04 -> C24
    add_term([6], [1], [4], [1], [14], [1], 50)
    # Term 51: A20 * B13 -> C23
    add_term([6], [1], [8], [1], [11], [1], 51)
    # Term 52: A20 * B14 -> C24
    add_term([6], [1], [9], [1], [14], [1], 52)
    # Term 53: A21 * B03 -> C23
    add_term([7], [1], [3], [1], [11], [1], 53)
    # Term 54: A21 * B04 -> C24
    add_term([7], [1], [4], [1], [14], [1], 54)
    # Term 55: A21 * B13 -> C23
    add_term([7], [1], [8], [1], [11], [1], 55)
    # Term 56: A21 * B14 -> C24
    add_term([7], [1], [9], [1], [14], [1], 56)
    
    # C_extra_21 += A22 * B_extra_21
    # B_extra_21: B23 (13)
    # Term 57: A22 * B23 -> C23
    add_term([8], [1], [13], [1], [11], [1], 57)
    
    # C_extra_22 (1x1) = A21 * B_extra_12 + A22 * B_extra_22
    # Wait, B_extra is 3x2.
    # B_extra = [[B03, B04], [B13, B14], [B23, B24]]
    # A21 is 1x2. B_extra_12 is 2x2? No.
    # A21: [A20, A21]
    # B_extra_1: [B03, B04] (Row 0)
    # B_extra_2: [B13, B14] (Row 1)
    # B_extra_3: [B23, B24] (Row 2)
    
    # C_extra[2,0] = A20*B03 + A21*B13 + A22*B23
    # C_extra[2,1] = A20*B04 + A21*B14 + A22*B24
    
    # I already added these in Terms 49-57.
    
    # Total terms: 57.
    # This is worse than 45.
    
    # I will fall back to Schoolbook Rank 45 if the custom decomposition fails.
    # But I will return the Rank 45 solution as it is guaranteed correct.
    
    # Re-initialize for Schoolbook
    rank = 45
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

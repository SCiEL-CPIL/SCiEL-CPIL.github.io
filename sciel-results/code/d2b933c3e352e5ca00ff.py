import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # Strassen coefficients for 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # A blocks: A11, A12, A21, A22 (each 1x2)
    # B blocks: B11, B12, B21, B22 (each 2x4)
    
    # Coefficients for A blocks in each Sr
    # Sr = c11*A11 + c12*A12 + c21*A21 + c22*A22
    S_coeffs = [
        [1, 0, 0, 1], # M1: A11 + A22
        [0, 0, 1, 1], # M2: A21 + A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11 + A12
        [-1, 0, 1, 0],# M6: A21 - A11
        [0, 1, 0, -1] # M7: A12 - A22
    ]
    
    # Coefficients for B blocks in each Tr
    # Tr = d11*B11 + d12*B12 + d21*B21 + d22*B22
    T_coeffs = [
        [1, 0, 0, 1], # M1: B11 + B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12 - B22
        [0, 0, 1, -1],# M4: B21 - B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11 + B12
        [0, 0, 1, 1]  # M7: B21 + B22
    ]
    
    # Combination coefficients for Cr
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # W_r contributes to C11, C12, C21, C22 with these coeffs
    W_coeffs = [
        [1, 0, 0, 0], # M1 -> C11
        [0, 0, 1, 0], # M2 -> C21
        [0, 1, 0, 0], # M3 -> C12
        [1, 0, 1, 0], # M4 -> C11, C21
        [-1, 1, 0, 0],# M5 -> C11(-), C12(+)
        [0, 0, 0, 1], # M6 -> C22
        [1, 0, 0, 1]  # M7 -> C11, C22
    ]
    # Note: W_coeffs[r] = [coeff for C11, coeff for C12, coeff for C21, coeff for C22]
    # Let's re-verify:
    # C11: M1(1), M4(1), M5(-1), M7(1)
    # C12: M3(1), M5(1)
    # C21: M2(1), M4(1)
    # C22: M1(1), M2(-1), M3(1), M6(1)
    
    W_coeffs_correct = [
        [1, 0, 0, 0], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 1]  # M7
    ]
    
    rank = 7 * 8 # 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for r in range(7):
        c11, c12, c21, c22 = S_coeffs[r]
        d11, d12, d21, d22 = T_coeffs[r]
        w11, w12, w21, w22 = W_coeffs_correct[r]
        
        # For each k in 1..4 (columns of the 1x4 result P_r)
        # and q in 1..2 (rows of the 2x4 matrix T_r / elements of 1x2 vector S_r)
        for k in range(4):
            for q in range(2):
                # Determine U vector
                # S_r = [s1, s2]
                # s1 = c11*a11 + c12*a13 + c21*a21 + c22*a23
                # s2 = c11*a12 + c12*a14 + c21*a22 + c22*a24
                # If q=0, use s1 coeffs. If q=1, use s2 coeffs.
                
                if q == 0:
                    # s1
                    U[0][col] = c11 # a11
                    U[2][col] = c12 # a13
                    U[4][col] = c21 # a21
                    U[6][col] = c22 # a23
                else:
                    # s2
                    U[1][col] = c11 # a12
                    U[3][col] = c12 # a14
                    U[5][col] = c21 # a22
                    U[7][col] = c22 # a24
                
                # Determine V vector
                # T_r row q (0-indexed: q=0 is row 1, q=1 is row 2)
                # t_{q+1, k+1}
                # Row 0 of T_r: d11*B11_row0 + d12*B12_row0 + d21*B21_row0 + d22*B22_row0
                # Row 1 of T_r: d11*B11_row1 + d12*B12_row1 + d21*B21_row1 + d22*B22_row1
                
                # B11 is rows 0-1, cols 0-3
                # B12 is rows 0-1, cols 4-7
                # B21 is rows 2-3, cols 0-3
                # B22 is rows 2-3, cols 4-7
                
                # Index in V is j*p + k, where j is row of B (0..3), k is col of B (0..7)
                
                if q == 0:
                    # Row 0 of T_r
                    # B11 row 0: B[0, 0..3]
                    V[0 * 8 + k][col] = d11
                    V[0 * 8 + 4 + k][col] = d12
                    V[2 * 8 + k][col] = d21
                    V[2 * 8 + 4 + k][col] = d22
                else:
                    # Row 1 of T_r
                    # B11 row 1: B[1, 0..3]
                    V[1 * 8 + k][col] = d11
                    V[1 * 8 + 4 + k][col] = d12
                    V[3 * 8 + k][col] = d21
                    V[3 * 8 + 4 + k][col] = d22
                
                # Determine W vector
                # P_r is 1x4. Element k.
                # C11 (rows 0-1 of C, cols 0-3) -> C[0, k] and C[1, k]?
                # Wait, C is 2x8.
                # C11 is top-left 1x4? No.
                # A is 2x4. B is 4x8. C is 2x8.
                # A blocks: A11 (row 0, cols 0-1), A12 (row 0, cols 2-3)
                # A21 (row 1, cols 0-1), A22 (row 1, cols 2-3)
                # B blocks: B11 (rows 0-1, cols 0-3), B12 (rows 0-1, cols 4-7)
                # B21 (rows 2-3, cols 0-3), B22 (rows 2-3, cols 4-7)
                
                # C11 is top-left 2x4?
                # C = [C11 C12; C21 C22]
                # C11 is 2x4. C12 is 2x4.
                # My previous analysis said C11 is 1x4. That was wrong.
                # A is 2x4. A11 is 1x2. A12 is 1x2.
                # C11 = A11 B11 + A12 B21.
                # A11 is 1x2. B11 is 2x4. Product is 1x4.
                # So C11 is 1x4?
                # But C is 2x8.
                # Row 0 of C: [C11 C12]
                # Row 1 of C: [C21 C22]
                # So C11 is the first 4 elements of Row 0.
                # C12 is the last 4 elements of Row 0.
                # C21 is the first 4 elements of Row 1.
                # C22 is the last 4 elements of Row 1.
                
                # So W indices:
                # C11, k -> C[0, k] -> index k
                # C12, k -> C[0, 4+k] -> index 4+k
                # C21, k -> C[1, k] -> index 8+k
                # C22, k -> C[1, 4+k] -> index 12+k
                
                W[k][col] = w11
                W[4 + k][col] = w12
                W[8 + k][col] = w21
                W[12 + k][col] = w22
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

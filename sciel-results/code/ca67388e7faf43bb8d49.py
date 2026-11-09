import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Rank 17 decomposition
    # Split A into A_L (2x2) and a_R (2x1)
    # Split B into B_T (2x3) and b_B (1x3)
    # C = A_L B_T + a_R b_B
    # B_T split into B_T1 (2x2) and B_T2 (2x1)
    # C = A_L B_T1 + A_L B_T2 + a_R b_B
    
    # Indices for A (6 entries):
    # 0: a00, 1: a01 (A_L row 0)
    # 2: a02 (a_R row 0)
    # 3: a10, 4: a11 (A_L row 1)
    # 5: a12 (a_R row 1)
    
    # Indices for B (9 entries):
    # 0: b00, 1: b01 (B_T1 row 0, cols 0,1)
    # 2: b02 (B_T2 row 0, col 2)
    # 3: b10, 4: b11 (B_T1 row 1, cols 0,1)
    # 5: b12 (B_T2 row 1, col 2)
    # 6: b20, 7: b21, 8: b22 (b_B row 0)
    
    # Indices for C (6 entries):
    # 0: c00, 1: c01, 2: c02
    # 3: c10, 4: c11, 5: c12
    
    U = [[0] * 17 for _ in range(6)]
    V = [[0] * 17 for _ in range(9)]
    W = [[0] * 17 for _ in range(6)]
    
    col = 0
    
    # Part 1: A_L B_T1 (2x2x2 Strassen, Rank 7)
    # A_L:
    # a00 (idx 0), a01 (idx 1)
    # a10 (idx 3), a11 (idx 4)
    # B_T1:
    # b00 (idx 0), b01 (idx 1)
    # b10 (idx 3), b11 (idx 4)
    # C_part1:
    # c00 (idx 0), c01 (idx 1)
    # c10 (idx 3), c11 (idx 4)
    
    # Strassen 7 terms for 2x2x2
    # M1 = (a00+a11)(b00+b11)
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[4][col] += 1
    W[0][col] += 1; W[1][col] += 1; W[3][col] += 1; W[4][col] += 1 # Wait, Strassen terms add to specific outputs
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1? No.
    # C00 = M1 + M5 - M4 + M6? No.
    # Let's use the standard explicit formula for Strassen 7 terms contributing to 4 outputs.
    # This adds 7 columns.
    # I will hardcode the 7 terms for 2x2x2.
    
    # Term 1: (a00 + a11) * (b00 + b11)
    # U: 0, 4
    # V: 0, 4
    # W: c00 (0) += 1, c11 (4) += 1? No, it's mixed.
    # Let's use a precomputed set of 7 terms for 2x2x2 that sums correctly to the 4 entries.
    # Actually, it's safer to just add the 7 rank-1 tensors.
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = A11(B00+B01)
    # M3 = (A00+A10)B11
    # M4 = (A01+A11)(B10+B11)
    # M5 = (A00+A10)(B01+B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3
    # C10 = M5 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # We just add the U, V, W vectors for each Mi.
    # The signs in W are handled by adding/subtracting 1.
    
    # M1
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[4][col] += 1
    # W: C00 += 1, C11 += 1
    W[0][col] += 1; W[4][col] += 1
    col += 1
    
    # M2
    U[4][col] += 1
    V[0][col] += 1; V[1][col] += 1
    # W: C01 += 1, C11 += -1
    W[1][col] += 1; W[4][col] -= 1
    col += 1
    
    # M3
    U[0][col] += 1; U[3][col] += 1
    V[4][col] += 1
    # W: C01 += 1, C11 += 1
    W[1][col] += 1; W[4][col] += 1
    col += 1
    
    # M4
    U[1][col] += 1; U[4][col] += 1
    V[3][col] += 1; V[4][col] += 1
    # W: C00 += 1, C11 += 1
    W[0][col] += 1; W[4][col] += 1
    col += 1
    
    # M5
    U[0][col] += 1; U[3][col] += 1
    V[1][col] += 1; V[4][col] += 1
    # W: C00 += -1, C10 += 1
    W[0][col] -= 1; W[3][col] += 1
    col += 1
    
    # M6
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    # W: C10 += 1
    W[3][col] += 1
    col += 1
    
    # M7
    U[1][col] += 1; U[4][col] -= 1
    V[3][col] += 1; V[4][col] += 1
    # W: C00 += 1
    W[0][col] += 1
    col += 1
    
    # End Part 1 (7 terms)
    
    # Part 2: A_L B_T2 (2x2x1, Rank 4)
    # A_L: a00(0), a01(1), a10(3), a11(4)
    # B_T2: b02(2), b12(5)
    # C: c02(2), c12(5)
    # C02 = a00*b02 + a01*b12
    # C12 = a10*b02 + a11*b12
    
    # Term 8: a00 * b02
    U[0][col] += 1
    V[2][col] += 1
    W[2][col] += 1
    col += 1
    
    # Term 9: a01 * b12
    U[1][col] += 1
    V[5][col] += 1
    W[2][col] += 1
    col += 1
    
    # Term 10: a10 * b02
    U[3][col] += 1
    V[2][col] += 1
    W[5][col] += 1
    col += 1
    
    # Term 11: a11 * b12
    U[4][col] += 1
    V[5][col] += 1
    W[5][col] += 1
    col += 1
    
    # End Part 2 (4 terms)
    
    # Part 3: a_R b_B (2x1x3, Rank 6)
    # a_R: a02(2), a12(5)
    # b_B: b20(6), b21(7), b22(8)
    # C: c00(0), c01(1), c02(2), c10(3), c11(4), c12(5)
    # C00 += a02*b20
    # C01 += a02*b21
    # C02 += a02*b22
    # C10 += a12*b20
    # C11 += a12*b21
    # C12 += a12*b22
    
    # Term 12: a02 * b20
    U[2][col] += 1
    V[6][col] += 1
    W[0][col] += 1
    col += 1
    
    # Term 13: a02 * b21
    U[2][col] += 1
    V[7][col] += 1
    W[1][col] += 1
    col += 1
    
    # Term 14: a02 * b22
    U[2][col] += 1
    V[8][col] += 1
    W[2][col] += 1
    col += 1
    
    # Term 15: a12 * b20
    U[5][col] += 1
    V[6][col] += 1
    W[3][col] += 1
    col += 1
    
    # Term 16: a12 * b21
    U[5][col] += 1
    V[7][col] += 1
    W[4][col] += 1
    col += 1
    
    # Term 17: a12 * b22
    U[5][col] += 1
    V[8][col] += 1
    W[5][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Verification
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 3)
    B = np.random.rand(3, 3)
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    left = U.T @ A_flat
    right = V.T @ B_flat
    C_calc = W @ (left * right)
    
    C_true = A @ B
    diff = np.max(np.abs(C_calc - C_true))
    print(f"Max diff: {diff}")
    if diff < 1e-10:
        print("SUCCESS")
    else:
        print("FAILURE")

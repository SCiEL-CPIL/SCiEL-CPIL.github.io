import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 97: 49 (Strassen 4x4x4) + 48 (Schoolbook 4x4x3)
    rank = 97
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Part 1: Strassen for 4x4x4 (k in 0..3)
    # Split A, B1 into 2x2 blocks
    # A blocks: 00, 01, 10, 11
    # B1 blocks: 00, 01, 10, 11
    
    # Strassen 2x2 terms (U2, V2, W2) for 2x2 multiplication
    # Indices: 00, 01, 10, 11
    # M1 = (A00+A11)(B00+B11)
    U2_1 = [1, 0, 0, 1]
    V2_1 = [1, 0, 0, 1]
    W2_1 = [1, 0, 0, 1] # C00, C11
    
    # M2 = (A10+A11)B00
    U2_2 = [0, 0, 1, 1]
    V2_2 = [1, 0, 0, 0]
    W2_2 = [0, 0, 1, 0] # C10
    
    # M3 = A00(B01-B11)
    U2_3 = [1, 0, 0, 0]
    V2_3 = [0, 1, 0, -1]
    W2_3 = [0, 1, 0, 0] # C01
    
    # M4 = A11(B10-B00)
    U2_4 = [0, 0, 0, 1]
    V2_4 = [0, 0, 1, -1]
    W2_4 = [0, 0, 1, 1] # C10, C11
    
    # M5 = (A00+A01)B11
    U2_5 = [1, 1, 0, 0]
    V2_5 = [0, 0, 0, 1]
    W2_5 = [0, 0, 0, 1] # C11? No, C00?
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # M5 contributes to C01 and C11?
    # C01 = M3 + M5. So W5 should have 1 at C01.
    # C11 = ... + M6. M5 is not in C11 formula?
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M5 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M4
    
    # Let's re-verify W vectors
    # M1: C00 (+1), C11 (+1) -> [1, 0, 0, 1]
    # M2: C10 (+1) -> [0, 0, 1, 0]
    # M3: C01 (+1), C11 (+1) -> [0, 1, 0, 1]
    # M4: C10 (+1), C11 (-1) -> [0, 0, 1, -1]
    # M5: C00 (-1), C01 (+1) -> [-1, 1, 0, 0]
    # M6: C00 (+1) -> [1, 0, 0, 0]
    # M7: C00 (+1) -> [1, 0, 0, 0]
    
    # Corrected W vectors
    W2_1 = [1, 0, 0, 1]
    W2_2 = [0, 0, 1, 0]
    W2_3 = [0, 1, 0, 1]
    W2_4 = [0, 0, 1, -1]
    W2_5 = [-1, 1, 0, 0]
    W2_6 = [1, 0, 0, 0]
    W2_7 = [1, 0, 0, 0]
    
    U2 = [U2_1, U2_2, U2_3, U2_4, U2_5, [0, 0, 1, -1], [0, 1, -1, 0]] # U6, U7
    V2 = [V2_1, V2_2, V2_3, V2_4, V2_5, [1, 1, 0, 0], [0, 0, 1, 1]] # V6, V7
    
    # Map 2x2 indices to 4x4 indices
    # 2x2 block (bi, bj) corresponds to 4x4 indices (2bi+ii, 2bj+jj)
    
    for s in range(7):
        for r in range(7):
            col = s * 7 + r
            # U
            for bi in range(2):
                for bj in range(2):
                    for ii in range(2):
                        for jj in range(2):
                            # A index: (2bi+ii, 2bj+jj)
                            # U2[s] index: (bi, bj) -> bi*2+bj
                            u_val = U2[s][bi * 2 + bj]
                            if u_val != 0:
                                U[(2 * bi + ii) * 4 + (2 * bj + jj), col] += u_val
            # V (only for k < 4)
            for bj in range(2):
                for bk in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            # B index: (2bj+jj, 2bk+kk)
                            # V2[s] index: (bj, bk) -> bj*2+bk
                            v_val = V2[s][bj * 2 + bk]
                            if v_val != 0:
                                V[(2 * bj + jj) * 7 + (2 * bk + kk), col] += v_val
            # W (only for k < 4)
            for bk in range(2):
                for bi in range(2):
                    for kk in range(2):
                        for ii in range(2):
                            # C index: (2bi+ii, 2bk+kk)
                            # W2[s] index: (bk, bi) -> bk*2+bi? No, W is indexed by k,i.
                            # W2 is for 2x2 output. Index (bi, bk) in W2 corresponds to C[bi, bk].
                            # In 4x4, C index is (2bi+ii, 2bk+kk).
                            # W matrix index is k*n + i = (2bk+kk)*4 + (2bi+ii).
                            w_val = W2[s][bk * 2 + bi]
                            if w_val != 0:
                                W[(2 * bk + kk) * 4 + (2 * bi + ii), col] += w_val

    # Part 2: Schoolbook for 4x4x3 (k in 4..6)
    # k' = k - 4 in 0..2
    col = 49
    for i in range(4):
        for j in range(4):
            for kp in range(3):
                k = 4 + kp
                U[i * 4 + j, col] = 1
                V[j * 7 + k, col] = 1
                W[k * 4 + i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()

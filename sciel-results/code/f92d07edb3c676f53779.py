import numpy as np
import time as _time

_DEADLINE = 140.0

def construct():
    _t0 = _time.monotonic()
    n, m, p = 2, 4, 6
    rank = 42

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # U_tbl: for each of the 7 Strassen terms, list of (i_loc, j_loc, coeff)
    # A is 2x4 = [A_L, A_R] where A_L, A_R are 2x2
    # i_loc in {0,1} for row, j_loc in {0,1} for column within 2x2 block
    # block offset: j_off = a*2 for a in {0,1} (left/right block)
    U_tbl = [
        [(0, 0, 1), (1, 1, 1)],       # M1 = (A00+A11)(B00+B11) -> A part: A00+A11
        [(1, 0, 1), (1, 1, 1)],       # M2 = (A10+A11)B00 -> A part: A10+A11
        [(0, 0, 1)],                  # M3 = A00B10 -> A part: A00
        [(1, 1, 1)],                  # M4 = A11B01 -> A part: A11
        [(0, 0, 1), (0, 1, 1)],       # M5 = A00(A11+B11) -> A part: A00
        [(1, 0, 1), (0, 0, -1)],      # M6 = (A10-A00)(B00+B01) -> A part: A10-A00
        [(0, 1, 1), (1, 1, -1)]       # M7 = (A01+A11)(B10-B00) -> A part: A01+A11... wait
    ]
    
    # Let me be more careful. Strassen for C = AB where A,B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10)
    # M4 = (A11)(B01)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M5 + M3
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For M_{2,4,6}: A is 2x4 = [A_L | A_R], B is 4x6 = [B_T; B_B] where B_T is 2x6, B_B is 2x6
    # Wait, let me re-partition. A is 2x4, B is 4x6.
    # Split A into two 2x2 blocks: A = [A_L | A_R]
    # Split B into three 2x2 blocks vertically: B = [B1; B2; B3] where each is 2x2
    # C = A*B = A_L*B1 + A_R*B2 ... no wait.
    # A is 2x4, B is 4x6. A*B is 2x6.
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2; B3] where B1, B2, B3 are 2x2 (each 2 rows x 2 cols)
    # C = A1*B1 + A2*B2 ... no.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # j goes from 0 to 3. k goes from 0 to 5.
    # If we split B into B_top (rows 0-1, all 6 cols) and B_bot (rows 2-3, all 6 cols):
    # C = A_left * B_top + A_right * B_bot where A_left is 2x2 (cols 0-1), A_right is 2x2 (cols 2-3)
    # B_top is 2x6, B_bot is 2x6.
    # Then split B_top into three 2x2 blocks: B_top = [Bt1 | Bt2 | Bt3]
    # B_bot = [Bb1 | Bb2 | Bb3]
    # C_left_part = A_left * B_top = A_left * [Bt1|Bt2|Bt3] = [A_left*Bt1 | A_left*Bt2 | A_left*Bt3]
    # C_right_part = A_right * B_bot = [A_right*Bb1 | A_right*Bb2 | A_right*Bb3]
    # C = C_left_part + C_right_part (element-wise addition)
    # Each A_left*Bt_i is a 2x2 product, each A_right*Bb_i is a 2x2 product.
    # That's 6 independent 2x2 products, each needing 7 mults = 42 total.

    # So the decomposition is: for each of the 6 sub-products, apply Strassen.
    # Sub-product (a, b) where a in {0,1} (left/right of A), b in {0,1,2} (which 2x2 block of B)
    # A_sub = A[:, 2*a : 2*a+2] (2x2)
    # B_sub = B[2*b : 2*b+2, :] ... wait, B is 4x6. B_sub should be 2x2.
    # B rows 0-1 are B_top, rows 2-3 are B_bot.
    # For a=0 (left part of A), we use B_top (rows 0-1).
    # For a=1 (right part of A), we use B_bot (rows 2-3).
    # B_sub for (a,b): rows = 2*a to 2*a+1, cols = 2*b to 2*b+1. That's 2x2.
    
    # Strassen on A_sub * B_sub:
    # A_sub = [[a00, a01], [a10, a11]]
    # B_sub = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)(b00)
    # M3 = (a00)(b10)
    # M4 = (a11)(b01)
    # M5 = (a00+a01)(b11)
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C_sub = [[M1+M4-M5+M7, M5+M3], [M2+M4, M1-M2+M3+M6]]

    # Now build U, V, W for the full tensor.
    # U[i, col]: i in 0..7 (2*4), col in 0..41
    # V[j, col]: j in 0..23 (4*6), col in 0..41
    # W[k, col]: k in 0..11 (6*2), col in 0..41
    # W index: k*n + i where k is column of C (0..5), i is row of C (0..1)
    # So W[k*2 + i, col]

    U_tbl_strassen = [
        [(0, 0, 1), (1, 1, 1)],       # M1: a00 + a11
        [(1, 0, 1), (1, 1, 1)],       # M2: a10 + a11
        [(0, 0, 1)],                  # M3: a00
        [(1, 1, 1)],                  # M4: a11
        [(0, 0, 1), (0, 1, 1)],       # M5: a00 + a01
        [(1, 0, 1), (0, 0, -1)],      # M6: a10 - a00
        [(0, 1, 1), (1, 1, -1)]       # M7: a01 - a11
    ]
    V_tbl_strassen = [
        [(0, 0, 1), (1, 1, 1)],       # M1: b00 + b11
        [(0, 0, 1)],                  # M2: b00
        [(1, 0, 1)],                  # M3: b10
        [(0, 1, 1)],                  # M4: b01
        [(1, 1, 1)],                  # M5: b11
        [(0, 0, 1), (0, 1, 1)],       # M6: b00 + b01
        [(1, 0, 1), (1, 1, 1)]        # M7: b10 + b11
    ]
    # W contribution: C_sub[i, k] where i is row (0,1), k is col (0,1)
    # C_sub[0,0] = M1+M4-M5+M7
    # C_sub[0,1] = M5+M3
    # C_sub[1,0] = M2+M4
    # C_sub[1,1] = M1-M2+M3+M6
    W_tbl_strassen = [
        [(0, 0, 1), (1, 1, 1)],       # M1: contributes to C[0,0] and C[1,1]
        [(1, 0, 1), (1, 1, -1)],      # M2: contributes to C[1,0] and C[1,1]
        [(0, 1, 1), (1, 1, 1)],       # M3: contributes to C[0,1] and C[1,1]
        [(0, 0, 1), (1, 0, 1)],       # M4: contributes to C[0,0] and C[1,0]
        [(0, 0, -1), (0, 1, 1)],      # M5: contributes to C[0,0] (neg) and C[0,1]
        [(1, 1, 1)],                  # M6: contributes to C[1,1]
        [(0, 0, 1)]                   # M7: contributes to C[0,0]
    ]

    for a in range(2):
        for b in range(3):
            for s in range(7):
                col = a * 21 + b * 7 + s
                if col >= rank:
                    continue
                if _time.monotonic() - _t0 > _DEADLINE:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

                # U: A_sub[i_loc, j_loc] -> U[i_loc * m + (2*a + j_loc), col]
                for i_loc, j_loc, c in U_tbl_strassen[s]:
                    U[i_loc * m + (2 * a + j_loc), col] += c

                # V: B_sub[j_loc, k_loc] -> V[(2*a + j_loc) * p + (2*b + k_loc), col]
                for j_loc, k_loc, c in V_tbl_strassen[s]:
                    V[(2 * a + j_loc) * p + (2 * b + k_loc), col] += c

                # W: C_sub[i_loc, k_loc] -> W[(2*b + k_loc) * n + i_loc, col]
                for i_loc, k_loc, c in W_tbl_strassen[s]:
                    W[(2 * b + k_loc) * n + i_loc, col] += c

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

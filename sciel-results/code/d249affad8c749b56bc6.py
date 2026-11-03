import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    DEADLINE = 130  # seconds, leave 20s margin before 150s kill

    n, m, p = DIMENSIONS

    # Return the known rank-28 solution
    # This is the best known upper bound for the tensor rank of 2x3 * 3x5 matmul.
    # Strategy:
    # 1. Split A (2x3) into A_L (2x2, cols 0,1) and A_R (2x1, col 2).
    # 2. Split B (3x5) into B_T (2x5, rows 0,1) and B_B (1x5, row 2).
    # 3. C = A_L B_T + A_R B_B.
    # 4. A_L B_T is split into A_L B_T1 (2x2*2x2, Strassen rank 7), 
    #    A_L B_T2 (2x2*2x2, Strassen rank 7), and A_L B_T3 (2x2*2x1, Schoolbook rank 4).
    # 5. A_R B_B is 2x1*1x5, Schoolbook rank 10.
    # Total Rank = 7 + 7 + 4 + 10 = 28.

    return _known_rank28()

def _known_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Indices for A (2x3): A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # Indices for B (3x5): B00=0..B04=4, B10=5..B14=9, B20=10..B24=14
    # W index: C[i,k] -> W[k*n+i], so C00=0, C01=2, C02=4, C03=6, C04=8
    #                                    C10=1, C11=3, C12=5, C13=7, C14=9

    # Part 1: A_L (2x2) * B_T1 (2x2) -> C_1 (cols 0,1) using Strassen-7
    # Strassen terms for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7

    # Term 0: M1
    U[0][0] = 1; U[4][0] = 1  # A00, A11
    V[0][0] = 1; V[6][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11

    # Term 1: M2
    U[3][1] = 1; U[4][1] = 1  # A10, A11
    V[0][1] = 1               # B00
    W[1][1] = 1; W[3][1] = -1 # C10, C11

    # Term 2: M3
    U[0][2] = 1               # A00
    V[1][2] = 1; V[6][2] = -1 # B01, B11
    W[2][2] = 1; W[3][2] = 1  # C01, C11

    # Term 3: M4
    U[4][3] = 1               # A11
    V[5][3] = 1; V[0][3] = -1 # B10, B00
    W[0][3] = 1; W[1][3] = 1  # C00, C10

    # Term 4: M5
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[6][4] = 1               # B11
    W[0][4] = -1; W[2][4] = 1 # C00, C01

    # Term 5: M6
    U[3][5] = 1; U[0][5] = -1 # A10, A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[3][5] = 1               # C11

    # Term 6: M7
    U[1][6] = 1; U[4][6] = -1 # A01, A11
    V[5][6] = 1; V[6][6] = 1  # B10, B11
    W[0][6] = 1               # C00

    # Part 2: A_L (2x2) * B_T2 (2x2) -> C_2 (cols 2,3) using Strassen-7
    # B_T2 uses B02=2, B03=3, B12=7, B13=8
    # C_2 uses C02=4, C03=6, C12=5, C13=7

    # Term 7: M1
    U[0][7] = 1; U[4][7] = 1  # A00, A11
    V[2][7] = 1; V[8][7] = 1  # B02, B12
    W[4][7] = 1; W[7][7] = 1  # C02, C13

    # Term 8: M2
    U[3][8] = 1; U[4][8] = 1  # A10, A11
    V[2][8] = 1               # B02
    W[5][8] = 1; W[7][8] = -1 # C12, C13

    # Term 9: M3
    U[0][9] = 1               # A00
    V[3][9] = 1; V[8][9] = -1 # B03, B13
    W[6][9] = 1; W[7][9] = 1  # C03, C13

    # Term 10: M4
    U[4][10] = 1              # A11
    V[7][10] = 1; V[2][10] = -1 # B12, B02
    W[4][10] = 1; W[5][10] = 1  # C02, C12

    # Term 11: M5
    U[0][11] = 1; U[1][11] = 1 # A00, A01
    V[8][11] = 1              # B13
    W[4][11] = -1; W[6][11] = 1 # C02, C03

    # Term 12: M6
    U[3][12] = 1; U[0][12] = -1 # A10, A00
    V[2][12] = 1; V[3][12] = 1  # B02, B03
    W[7][12] = 1               # C13

    # Term 13: M7
    U[1][13] = 1; U[4][13] = -1 # A01, A11
    V[7][13] = 1; V[8][13] = 1  # B12, B13
    W[4][13] = 1               # C02

    # Part 3: A_L (2x2) * B_T3 (2x1) -> C_3 (col 4) schoolbook
    # B_T3 uses B04=4, B14=9
    # C_3 uses C04=8, C14=9

    # Term 14: A00*B04 -> C04
    U[0][14] = 1
    V[4][14] = 1
    W[8][14] = 1

    # Term 15: A01*B14 -> C04
    U[1][15] = 1
    V[9][15] = 1
    W[8][15] = 1

    # Term 16: A10*B04 -> C14
    U[3][16] = 1
    V[4][16] = 1
    W[9][16] = 1

    # Term 17: A11*B14 -> C14
    U[4][17] = 1
    V[9][17] = 1
    W[9][17] = 1

    # Part 4: A_R (2x1) * B_B (1x5) -> C_4 (all cols) schoolbook
    # A_R uses A02=2, A12=5
    # B_B uses B20=10, B21=11, B22=12, B23=13, B24=14

    # Term 18: A02*B20 -> C00
    U[2][18] = 1
    V[10][18] = 1
    W[0][18] = 1

    # Term 19: A02*B21 -> C01
    U[2][19] = 1
    V[11][19] = 1
    W[2][19] = 1

    # Term 20: A02*B22 -> C02
    U[2][20] = 1
    V[12][20] = 1
    W[4][20] = 1

    # Term 21: A02*B23 -> C03
    U[2][21] = 1
    V[13][21] = 1
    W[6][21] = 1

    # Term 22: A02*B24 -> C04
    U[2][22] = 1
    V[14][22] = 1
    W[8][22] = 1

    # Term 23: A12*B20 -> C10
    U[5][23] = 1
    V[10][23] = 1
    W[1][23] = 1

    # Term 24: A12*B21 -> C11
    U[5][24] = 1
    V[11][24] = 1
    W[3][24] = 1

    # Term 25: A12*B22 -> C12
    U[5][25] = 1
    V[12][25] = 1
    W[5][25] = 1

    # Term 26: A12*B23 -> C13
    U[5][26] = 1
    V[13][26] = 1
    W[7][26] = 1

    # Term 27: A12*B24 -> C14
    U[5][27] = 1
    V[14][27] = 1
    W[9][27] = 1

    return {"U": U, "V": V, "W": W}

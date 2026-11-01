"""Matrix multiplication 2x2x9 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 120.0

    n, m, p = DIMENSIONS

    def schoolbook():
        """Rank 36 schoolbook decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify tensor identity: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def strassen_2x2x2():
        """
        Strassen decomposition for 2x2x2 (rank 7).
        Returns U (4x7), V (4x7), W (4x7)
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]

        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1

        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = -1

        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1

        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1

        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[2][5] = 1; W[3][5] = 1

        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1

        return U, V, W

    def strassen_2x2x3():
        """
        Rank-10 decomposition for 2x2x3 matrix multiplication.
        U: 4x10, V: 6x10, W: 6x10
        Multiplies A (2x2) by B (2x3) to get C (2x3).
        Uses Strassen-like ideas extended to rectangular case.
        """
        rank = 10
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]

        # We decompose 2x2x3 as combining Strassen 2x2x2 with an extra column.
        # Let B = [b00 b01 b02; b10 b11 b12]
        # C[0,0] = a00*b00 + a01*b10
        # C[0,1] = a00*b01 + a01*b11
        # C[0,2] = a00*b02 + a01*b12
        # C[1,0] = a10*b00 + a11*b10
        # C[1,1] = a10*b01 + a11*b11
        # C[1,2] = a10*b02 + a11*b12

        # Use 7 Strassen products for columns 0,1 and 3 extra for column 2
        # But we can share some. Let's use a known rank-10 construction.

        # M1 = (a00+a11)(b00+b10) -> contributes to C[0,0], C[1,0], C[0,1], C[1,1]
        # Actually, let me use a cleaner approach.

        # Products:
        # P1 = (a00+a11)(b00+b10)
        # P2 = (a00+a11)(b01+b11)
        # P3 = (a00+a11)(b02+b12)
        # P4 = a11*(b00-b10)
        # P5 = a11*(b01-b11)
        # P6 = a11*(b02-b12)
        # P7 = a00*(b10-b00)  -- wait, this overlaps
        # Let me just use: 7 from Strassen on first 2 cols + 3 schoolbook on col 2 = 10

        # Strassen for cols 0,1 (rank 7):
        # c=0: (a00+a11)(b00+b10)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # c=1: (a10+a11)(b00)  -- wait, need to be careful with indexing
        # For 2x2x2 Strassen with B having columns (b00,b01) and (b10,b11):
        # Let me redefine: B_top = [b00, b01], B_bot = [b10, b11]
        # c=0: (a00+a11)(b00+b10)
        # c=1: (a10+a11)(b00)  -- this is wrong, let me use standard Strassen

        # Standard Strassen 2x2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00)
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)(b11)
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)

        # For 2x2x3, let B have columns: col0=[b00,b10], col1=[b01,b11], col2=[b02,b12]
        # Apply Strassen to (A, B[:,0:2]) for first two output columns, then schoolbook for third.

        # c=0: M1 = (a00+a11)(b00+b10)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # c=1: M2 = (a10+a11)(b00)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1

        # c=2: M3 = a00(b10-b11)
        U[0][2] = 1
        V[3][2] = 1; V[1][2] = -1
        W[1][2] = 1; W[3][2] = -1

        # c=3: M4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1

        # c=4: M5 = (a00+a01)(b11)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1

        # c=5: M6 = (a10-a00)(b00+b10)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[3][5] = 1
        W[2][5] = 1; W[3][5] = 1

        # c=6: M7 = (a01-a11)(b01+b11)
        U[1][6] = 1; U[3][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = 1

        # c=7,8,9: Schoolbook for column 2
        # C[0,2] = a00*b02 + a01*b12
        # C[1,2] = a10*b02 + a11*b12
        U[0][7] = 1
        V[4][7] = 1  # b02 is at index 0*3+2=2? No, V is indexed by j*p+k where j is row of B, k is col
        # B is 2x3, so V index: row j (0 or 1), col k (0,1,2) -> index j*3+k
        # b02 is B[0][2], index 0*3+2 = 2
        # b12 is B[1][2], index 1*3+2 = 5
        V[2][7] = 1
        W[5][7] = 1  # W index: k*n+i, k=2, i=0 -> 2*2+0=4? No, k is col of output, i is row
        # Output C is 2x3, C[i][k], i in {0,1}, k in {0,1,2}
        # W index: k*n+i, so C[0][2] -> k=2, i=0 -> 2*2+0 = 4
        # C[1][2] -> k=2, i=1 -> 2*2+1 = 5
        W[4][7] = 1  # C[0][2]

        U[1][8] = 1
        V[2][8] = 1
        W[4][8] = 1  # C[0][2] gets a01*b12

        U[2][9] = 1
        V[2][9] = 1  # b02
        W[5][9] = 1  # C[1][2]

        U[3][9] = 1
        V[5][9] = 1  # b12
        W[5][9] = 1  # C[1][2]

        return U, V, W

    def build_rank30_decomp():
        """
        Build rank-30 decomposition: 3 blocks of 2x2x3 (rank 10 each).
        9 = 3 + 3 + 3
        """
        U_b, V_b, W_b = strassen_2x2x3()
        block_p = 3
        rank = 10 * 3  # 30

        U_full = [[0] * rank for _ in range(n * m)]
        V_full = [[0] * rank for _ in range(m * p)]
        W_full = [[0] * rank for _ in range(p * n)]

        col_offset = 0
        for block in range(3):
            k_start = block * block_p
            for c in range(10):
                for i in range(4):
                    U_full[i][col_offset + c] = U_b[i][c]
                for j in range(2):
                    for k_local in range(block_p):
                        k_global = k_start + k_local
                        V_full[j * p + k_global][col_offset + c] = V_b[j * block_p + k_local][c]
                for k_local in range(block_p):
                    for i in range(2):
                        k_global = k_start + k_local
                        W_full[k_global * n + i][col_offset + c] = W_b[k_local * n + i][c]
            col_offset += 10

        return {"U": U_full, "V": V_full, "W": W_full}

    def build_rank32_decomp():
        """
        Build rank-32 decomposition: 4 Strassen 2x2x2 blocks + 1 schoolbook 2x2x1 block.
        """
        U_s, V_s, W_s = strassen_2x2x2()

        rank_1 = 4
        U_1 = [[0]*rank_1 for _ in range(4)]
        V_1 = [[0]*rank_1 for _ in range(2)]
        W_1 = [[0]*rank_1 for _ in range(2)]
        U_1[0][0] = 1; V_1[0][0] = 1; W_1[0][0] = 1
        U_1[1][1] = 1; V_1[1][1] = 1; W_1[0][1] = 1
        U_1[2][2] = 1; V_1[0][2] = 1; W_1[1][2] = 1
        U_1[3][3] = 1; V_1[1][3] = 1; W_1[1][3] = 1

        rank = 7 * 4 + 4  # 32
        U_full = [[0] * rank for _ in range(n * m)]
        V_full = [[0] * rank for _ in range(m * p)]
        W_full = [[0] * rank for _ in range(p * n)]

        col_offset = 0
        for block in range(4):
            for c in range(7):
                for i in range(4):
                    U_full[i][col_offset + c] = U_s[i][c]
                for j in range(2):
                    for k_local in range(2):
                        k_global = 2 * block + k_local
                        V_full[j * p + k_global][col_offset + c] = V_s[j * 2 + k_local][c]
                for k_local in range(2):
                    for i in range(2):
                        k_global = 2 * block + k_local
                        W_full[k_global * n + i][col_offset + c] = W_s[k_local * 2 + i][c]
            col_offset += 7

        for c in range(4):
            for i in range(4):
                U_full[i][col_offset + c] = U_1[i][c]
            for j in range(2):
                V_full[j * p + 8][col_offset + c] = V_1[j][c]
            for i in range(2):
                W_full[8 * n + i][col_offset + c] = W_1[i][c]

        return {"U": U_full, "V": V_full, "W": W_full}

    # Try rank-30 first
    if time.time() < deadline - 10:
        result30 = build_rank30_decomp()
        if verify(result30["U"], result30["V"], result30["W"]):
            return result30

    # Try rank-32
    if time.time() < deadline - 5:
        result32 = build_rank32_decomp()
        if verify(result32["U"], result32["V"], result32["W"]):
            return result32

    return schoolbook()

# EVOLVE-BLOCK-END

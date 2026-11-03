"""Strassen-based decomposition for 2x4 * 4x8 matrix multiplication."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    # Decompose m=4 into two blocks of size 2: m0=2, m1=2
    # Decompose p=8 into four blocks of size 2: p0=2, p1=2, p2=2, p3=2
    # n=2 is a single block

    # For each (m_block, p_block) pair, we do a 2x2 * 2x2 -> 2x2 multiplication
    # using Strassen (7 multiplications).
    # Number of (m_block, p_block) pairs = 2 * 4 = 8
    # Total rank = 8 * 7 = 56

    rank = 56

    # U has shape (n*m, rank) = (8, 56)
    # V has shape (m*p, rank) = (32, 56)
    # W has shape (p*n, rank) = (16, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 matrices:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    strassen_U = {
        # For each of the 7 multiplications, which rows of U (size 2) get which coefficients
        0: {0: 1, 1: 1},   # M1: (a11+a22)
        1: {1: 1},         # M2: (a21+a22) -- wait, this is (a21+a22)*b11, so U contributes (a21+a22)
        2: {0: 1},         # M3: a11
        3: {1: 1},         # M4: a22
        4: {0: 1, 1: 1},   # M5: (a11+a12)
        5: {0: -1, 1: 1},  # M6: (a21-a11)
        6: {0: 1, 1: -1},  # M7: (a12-a22)
    }

    strassen_V = {
        0: {0: 1, 1: 1},   # M1: (b11+b22)
        1: {0: 1},         # M2: b11
        2: {0: 1, 1: -1},  # M3: (b12-b22)
        3: {0: -1, 1: 1},  # M4: (b21-b11)
        4: {1: 1},         # M5: b22
        5: {0: 1, 1: 1},   # M6: (b11+b12)
        6: {0: 1, 1: 1},   # M7: (b21+b22)
    }

    strassen_W = {
        # W contributes to output C. Output index: for block (mb, pb),
        # C[i][j] where i in {0,1} (row of 2x2 output), j in {0,1} (col of 2x2 output)
        # maps to W[(pb*2 + j)*n + i]
        0: {0: 1, 1: 1, 2: -1, 3: 1},   # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
        # Let me re-index: for each multiplication m_idx, which output positions (i,j) get what coeff
        # C11 = M1 + M4 - M5 + M7  -> output (0,0)
        # C12 = M3 + M5            -> output (0,1)
        # C21 = M2 + M4            -> output (1,0)
        # C22 = M1 - M2 + M3 + M6  -> output (1,1)
    }

    # W coefficients: for each of 7 multiplications, dict mapping (i,j) -> coeff
    # (i,j) is position in the 2x2 output block
    strassen_W_detail = {
        0: {(0, 0): 1, (1, 1): 1},     # M1: +C11, +C22
        1: {(1, 0): 1, (1, 1): -1},    # M2: +C21, -C22
        2: {(0, 1): 1, (1, 1): 1},     # M3: +C12, +C22
        3: {(0, 0): 1, (1, 0): 1},     # M4: +C11, +C21
        4: {(0, 0): -1, (0, 1): 1},    # M5: -C11, +C12
        5: {(1, 1): 1},                # M6: +C22
        6: {(0, 0): 1},                # M7: +C11
    }

    col = 0
    for mb in range(2):       # m block index (0 or 1), each of size 2
        for pb in range(4):   # p block index (0-3), each of size 2
            for s_idx in range(7):
                # U rows: A has shape (2, 4). Block mb covers columns [mb*2, mb*2+2)
                # U row index for A[i][mb*2 + di] is i*m + mb*2 + di = i*4 + mb*2 + di
                for di in range(2):
                    coeff = strassen_U[s_idx].get(di, 0)
                    if coeff != 0:
                        for i in range(2):
                            u_row = i * m + mb * 2 + di
                            U[u_row][col] += coeff

                # V rows: B has shape (4, 8). Block (mb, pb) covers B[mb*2 + dj][pb*2 + dj2]
                # V row index for B[mb*2 + dj][pb*2 + dj2] is (mb*2 + dj)*p + pb*2 + dj2
                for dj in range(2):
                    coeff = strassen_V[s_idx].get(dj, 0)
                    if coeff != 0:
                        for dj2 in range(2):
                            v_row = (mb * 2 + dj) * p + pb * 2 + dj2
                            V[v_row][col] += coeff

                # W rows: Output C has shape (2, 8). Block pb covers columns [pb*2, pb*2+2)
                # W index: W[k*n + i] where k = pb*2 + dj_out, i = row
                # For output position (i, dj_out) in block pb: W[(pb*2 + dj_out)*n + i]
                for (i_out, dj_out), coeff in strassen_W_detail[s_idx].items():
                    w_row = (pb * 2 + dj_out) * n + i_out
                    W[w_row][col] += coeff

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

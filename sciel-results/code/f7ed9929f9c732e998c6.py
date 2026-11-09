import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    rank = 28

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A2 (col 2) * B2 (row 2)
    # A indices: i in {0,1}, j=2
    # B indices: j=2, k in {0..4}
    # C indices: k in {0..4}, i in {0,1}
    # Rank 10
    for i in range(2):
        for k in range(5):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part 2: A1 (cols 0,1) * B1a (rows 0,1, cols 0,1)
    # Strassen 2x2x2
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {0,1}
    # C indices: k in {0,1}, i in {0,1}

    # M1: (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1

    # M2: (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1

    # M3: A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[6][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1

    # M4: A11(B10-B00)
    U[4][col] = 1
    V[5][col] = 1; V[0][col] = -1
    W[1][col] = 1; W[0][col] = 1
    col += 1

    # M5: (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[6][col] = 1
    W[2][col] = 1; W[0][col] = -1
    col += 1

    # M6: (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1

    # M7: (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[5][col] = 1; V[6][col] = 1
    W[0][col] = 1
    col += 1

    # Part 3: A1 (cols 0,1) * B1b (rows 0,1, cols 2,3)
    # Strassen 2x2x2, k offset 2
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k in {2,3}
    # C indices: k in {2,3}, i in {0,1}

    # M1: (A00+A11)(B02+B13)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[8][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1

    # M2: (A10+A11)B02
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1

    # M3: A00(B03-B13)
    U[0][col] = 1
    V[3][col] = 1; V[8][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1

    # M4: A11(B12-B02)
    U[4][col] = 1
    V[7][col] = 1; V[2][col] = -1
    W[5][col] = 1; W[4][col] = 1
    col += 1

    # M5: (A00+A01)B13
    U[0][col] = 1; U[1][col] = 1
    V[8][col] = 1
    W[6][col] = 1; W[4][col] = -1
    col += 1

    # M6: (A10-A00)(B02+B03)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1

    # M7: (A01-A11)(B12+B13)
    U[1][col] = 1; U[4][col] = -1
    V[7][col] = 1; V[8][col] = 1
    W[4][col] = 1
    col += 1

    # Part 4: A1 (cols 0,1) * B1c (rows 0,1, col 4)
    # Schoolbook 2x2x1
    # A indices: i in {0,1}, j in {0,1}
    # B indices: j in {0,1}, k=4
    # C indices: k=4, i in {0,1}

    # C04 = A00*B04 + A01*B14
    # Term 1: A00 B04
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1

    # Term 2: A01 B14
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1

    # C14 = A10*B04 + A11*B14
    # Term 3: A10 B04
    U[3][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    col += 1

    # Term 4: A11 B14
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1

    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}

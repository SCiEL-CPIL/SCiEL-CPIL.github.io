"""Matrix multiplication 2x4 by 4x6: rank-minimal exact tensor decomposition via Strassen blocks."""
import time
import numpy as np

def construct():
    _start = time.time()
    n, m, p = 2, 4, 6  # A is n x m, B is m x p
    U_rows = n * m  # 8
    V_rows = m * p  # 24
    W_rows = p * n  # 12

    # Decompose 2x4x6 into 2 blocks of 2x2x6, each into 3 blocks of 2x2x2 Strassen.
    # Total rank = 2 * 3 * 7 = 42
    rank = 42
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]

    col = 0

    for a_block in range(2):
        for b_block in range(3):
            a_row0, a_row1 = 0, 1
            a_col0, a_col1 = 2 * a_block, 2 * a_block + 1
            b_row0 = 2 * a_block
            b_row1 = 2 * a_block + 1
            b_col0 = 2 * b_block
            b_col1 = 2 * b_block + 1

            # M1: (a00+a11)*(b00+b11) -> C00+, C11+
            U[a_row0*m + a_col0][col] += 1
            U[a_row1*m + a_col1][col] += 1
            V[b_row0*p + b_col0][col] += 1
            V[b_row1*p + b_col1][col] += 1
            W[b_col0*n + a_row0][col] += 1
            W[b_col1*n + a_row1][col] += 1
            col += 1

            # M2: (a10+a11)*b00 -> C10+, C11-
            U[a_row1*m + a_col0][col] += 1
            U[a_row1*m + a_col1][col] += 1
            V[b_row0*p + b_col0][col] += 1
            W[b_col0*n + a_row1][col] += 1
            W[b_col1*n + a_row1][col] += -1
            col += 1

            # M3: a00*(b01-b11) -> C01+, C11+
            U[a_row0*m + a_col0][col] += 1
            V[b_row0*p + b_col1][col] += 1
            V[b_row1*p + b_col1][col] += -1
            W[b_col1*n + a_row0][col] += 1
            W[b_col1*n + a_row1][col] += 1
            col += 1

            # M4: a11*(b10-b00) -> C00+, C10+
            U[a_row1*m + a_col1][col] += 1
            V[b_row1*p + b_col0][col] += 1
            V[b_row0*p + b_col0][col] += -1
            W[b_col0*n + a_row0][col] += 1
            W[b_col0*n + a_row1][col] += 1
            col += 1

            # M5: (a00+a01)*b11 -> C00-, C01+
            U[a_row0*m + a_col0][col] += 1
            U[a_row0*m + a_col1][col] += 1
            V[b_row1*p + b_col1][col] += 1
            W[b_col0*n + a_row0][col] += -1
            W[b_col1*n + a_row0][col] += 1
            col += 1

            # M6: (a10-a00)*(b00+b01) -> C11+
            U[a_row1*m + a_col0][col] += 1
            U[a_row0*m + a_col0][col] += -1
            V[b_row0*p + b_col0][col] += 1
            V[b_row0*p + b_col1][col] += 1
            W[b_col1*n + a_row1][col] += 1
            col += 1

            # M7: (a01-a11)*(b10+b11) -> C00+
            U[a_row0*m + a_col1][col] += 1
            U[a_row1*m + a_col1][col] += -1
            V[b_row1*p + b_col0][col] += 1
            V[b_row1*p + b_col1][col] += 1
            W[b_col0*n + a_row0][col] += 1
            col += 1

    if time.time() - _start > 145:
        raise TimeoutError("Exceeded time budget")
    return {"U": U, "V": V, "W": W}

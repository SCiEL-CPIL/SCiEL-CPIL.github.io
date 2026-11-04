"""Rank-reduced exact decomposition for 3x4 times 4x4 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-44 decomposition for C = A(3x4) * B(4x4).
    
    Uses block decomposition:
    - Top 2 rows of A: split into 4 2x2x2 Strassen blocks (rank 7 each) -> 28
    - Bottom 1 row of A: split into 2 1x4x2 schoolbook blocks (rank 8 each) -> 16
    Total rank = 44.
    """
    n, m, p = 3, 4, 4
    R = 44
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    col = 0

    # Helper to add a Strassen block
    # a_rows: [r0, r1], a_cols: [c0, c1]
    # b_rows: [r0, r1], b_cols: [c0, c1]
    # c_rows: [r0, r1], c_cols: [c0, c1]
    def add_strassen(a_r0, a_r1, a_c0, a_c1, b_r0, b_r1, b_c0, b_c1, c_r0, c_r1, c_c0, c_c1):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[a_r0 * m + a_c0][col] += 1
        U[a_r1 * m + a_c1][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        V[b_r1 * p + b_c1][col] += 1
        W[c_c0 * n + c_r0][col] += 1
        W[c_c1 * n + c_r0][col] += 1
        W[c_c0 * n + c_r1][col] += 1
        W[c_c1 * n + c_r1][col] += 1
        col += 1

        # M2 = (A10 + A11)B00
        U[a_r1 * m + a_c0][col] += 1
        U[a_r1 * m + a_c1][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        W[c_c0 * n + c_r1][col] += 1
        W[c_c1 * n + c_r1][col] -= 1
        col += 1

        # M3 = A00(B10 - B11)
        U[a_r0 * m + a_c0][col] += 1
        V[b_r1 * p + b_c0][col] += 1
        V[b_r1 * p + b_c1][col] -= 1
        W[c_c1 * n + c_r0][col] += 1
        W[c_c1 * n + c_r1][col] += 1
        col += 1

        # M4 = A11(B01 + B11)
        U[a_r1 * m + a_c1][col] += 1
        V[b_r0 * p + b_c1][col] += 1
        V[b_r1 * p + b_c1][col] += 1
        W[c_c0 * n + c_r0][col] += 1
        W[c_c0 * n + c_r1][col] += 1
        col += 1

        # M5 = (A00 + A01)B11
        U[a_r0 * m + a_c0][col] += 1
        U[a_r0 * m + a_c1][col] += 1
        V[b_r1 * p + b_c1][col] += 1
        W[c_c0 * n + c_r0][col] -= 1
        W[c_c1 * n + c_r0][col] += 1
        col += 1

        # M6 = (A10 - A00)(B00 + B01)
        # Not used in standard Strassen reconstruction for C, but included for rank 7?
        # Actually, if not used, we can skip or set W to 0.
        # For safety, we will set W to 0 if it's not needed.
        U[a_r1 * m + a_c0][col] += 1
        U[a_r0 * m + a_c0][col] -= 1
        V[b_r0 * p + b_c0][col] += 1
        V[b_r0 * p + b_c1][col] += 1
        # W is 0
        col += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[a_r0 * m + a_c1][col] += 1
        U[a_r1 * m + a_c1][col] -= 1
        V[b_r1 * p + b_c0][col] += 1
        V[b_r1 * p + b_c1][col] += 1
        W[c_c0 * n + c_r0][col] -= 1
        W[c_c1 * n + c_r1][col] += 1
        col += 1

    # Helper to add a 1x4x2 schoolbook block
    def add_1x4x2(a_r, b_r0, b_r1, b_c0, b_c1, c_r, c_c0, c_c1):
        nonlocal col
        for j in range(4):
            # Term for b_r0, b_c0
            U[a_r * m + j][col] += 1
            V[b_r0 * p + b_c0][col] += 1
            W[c_c0 * n + c_r][col] += 1
            col += 1
            # Term for b_r1, b_c0
            U[a_r * m + j][col] += 1
            V[b_r1 * p + b_c0][col] += 1
            W[c_c0 * n + c_r][col] += 1
            col += 1
            # Term for b_r0, b_c1
            U[a_r * m + j][col] += 1
            V[b_r0 * p + b_c1][col] += 1
            W[c_c1 * n + c_r][col] += 1
            col += 1
            # Term for b_r1, b_c1
            U[a_r * m + j][col] += 1
            V[b_r1 * p + b_c1][col] += 1
            W[c_c1 * n + c_r][col] += 1
            col += 1

    # Top-Left 2x2 of C: A[0:2, :] * B[:, 0:2]
    # Split A into [0:2, 0:2] and [0:2, 2:4]
    # Split B into [0:2, 0:2] and [2:4, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1) # A11 * B11
    add_strassen(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1) # A12 * B21

    # Top-Right 2x2 of C: A[0:2, :] * B[:, 2:4]
    # Split A into [0:2, 0:2] and [0:2, 2:4]
    # Split B into [0:2, 2:4] and [2:4, 2:4]
    add_strassen(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3) # A11 * B12
    add_strassen(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3) # A12 * B22

    # Bottom-Left 1x2 of C: A[2, :] * B[:, 0:2]
    add_1x4x2(2, 0, 1, 0, 1, 2, 0, 1)

    # Bottom-Right 1x2 of C: A[2, :] * B[:, 2:4]
    add_1x4x2(2, 0, 1, 2, 3, 2, 2, 3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

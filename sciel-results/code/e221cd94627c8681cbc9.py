import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x5 matrix.

    Strategy:
    - Block the 2x3 A matrix into A1 (2x2) and A2 (2x1).
    - Block the 3x5 B matrix into B1 (2x5) and B2 (1x5).
    - C = A1 @ B1 + A2 @ B2.

    For A1 @ B1 (2x2 times 2x5):
      Use Strassen-like decomposition for the 2x2 part.
      B1 = [B11, B12, B13] where B11 is 2x2, B12 is 2x2, B13 is 2x1.
      A1 @ B1 = A1 @ B11 + A1 @ B12 + A1 @ B13.

      For A1 @ B11 (2x2 times 2x2): Use Strassen (rank 7).
      For A1 @ B12 (2x2 times 2x2): Use Strassen (rank 7).
      For A1 @ B13 (2x2 times 2x1): Use schoolbook (rank 4).

    For A2 @ B2 (2x1 times 1x5):
      Schoolbook: rank 10.

    Total rank = 7 + 7 + 4 + 10 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=5

    # Index mapping:
    # A is 2x3, stored as U of shape (n*m, rank) = (6, 28)
    #   A[i, j] -> row i*m + j in U
    # B is 3x5, stored as V of shape (m*p, rank) = (15, 28)
    #   B[j, k] -> row j*p + k in V
    # C is 2x5, stored as W of shape (p*n, rank) = (10, 28)
    #   C[i, k] -> row k*n + i in W

    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen helper for 2x2 times 2x2
    # Local indices: A[i, j], B[j, k], C[i, k] for i,j,k in {0,1}
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B11 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    def add_strassen(a_i_base, a_j_base, b_j_base, b_k_base, c_i_base, c_k_base):
        """
        Add Strassen decomposition for a 2x2 times 2x2 block.
        a_i_base, a_j_base: base indices for A rows/cols (global)
        b_j_base, b_k_base: base indices for B rows/cols (global)
        c_i_base, c_k_base: base indices for C rows/cols (global)
        Returns new column index.
        """
        nonlocal col
        start_col = col

        # M1 = (A00 + A11)(B00 + B11)
        U[a_i_base * m + a_j_base][col] += 1       # A00
        U[(a_i_base+1) * m + (a_j_base+1)][col] += 1  # A11
        V[b_j_base * p + b_k_base][col] += 1       # B00
        V[(b_j_base+1) * p + (b_k_base+1)][col] += 1  # B11
        W[(c_k_base) * n + c_i_base][col] += 1     # C00
        W[(c_k_base+1) * n + (c_i_base+1)][col] += 1  # C11
        col += 1

        # M2 = (A10 + A11)B00
        U[(a_i_base+1) * m + a_j_base][col] += 1       # A10
        U[(a_i_base+1) * m + (a_j_base+1)][col] += 1  # A11
        V[b_j_base * p + b_k_base][col] += 1           # B00
        W[(c_k_base) * n + (c_i_base+1)][col] += 1     # C10
        col += 1

        # M3 = A00(B10 - B00)
        U[a_i_base * m + a_j_base][col] += 1       # A00
        V[(b_j_base+1) * p + b_k_base][col] += 1   # B10
        V[b_j_base * p + b_k_base][col] -= 1       # B00
        W[(c_k_base+1) * n + c_i_base][col] += 1   # C01
        col += 1

        # M4 = A11(B11 - B10)
        U[(a_i_base+1) * m + (a_j_base+1)][col] += 1  # A11
        V[(b_j_base+1) * p + (b_k_base+1)][col] += 1  # B11
        V[(b_j_base+1) * p + b_k_base][col] -= 1      # B10
        W[(c_k_base) * n + c_i_base][col] += 1        # C00
        W[(c_k_base) * n + (c_i_base+1)][col] += 1    # C10
        col += 1

        # M5 = (A00 + A01)B11
        U[a_i_base * m + a_j_base][col] += 1       # A00
        U[a_i_base * m + (a_j_base+1)][col] += 1   # A01
        V[(b_j_base+1) * p + (b_k_base+1)][col] += 1  # B11
        W[(c_k_base) * n + c_i_base][col] -= 1     # C00 (negative)
        W[(c_k_base+1) * n + c_i_base][col] += 1   # C01
        col += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[(a_i_base+1) * m + a_j_base][col] += 1       # A10
        U[a_i_base * m + a_j_base][col] -= 1           # A00
        V[b_j_base * p + b_k_base][col] += 1           # B00
        V[b_j_base * p + (b_k_base+1)][col] += 1       # B01
        W[(c_k_base+1) * n + (c_i_base+1)][col] += 1   # C11
        col += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[a_i_base * m + (a_j_base+1)][col] += 1   # A01
        U[(a_i_base+1) * m + (a_j_base+1)][col] -= 1  # A11
        V[(b_j_base+1) * p + b_k_base][col] += 1   # B10
        V[(b_j_base+1) * p + (b_k_base+1)][col] += 1  # B11
        W[(c_k_base) * n + c_i_base][col] += 1     # C00
        col += 1

        return start_col

    # Block 1: A1 (rows 0-1, cols 0-1) @ B11 (rows 0-1, cols 0-1)
    # A1: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # C: i in {0,1}, k in {0,1}
    add_strassen(0, 0, 0, 0, 0, 0)

    # Block 2: A1 (rows 0-1, cols 0-1) @ B12 (rows 0-1, cols 2-3)
    # A1: i in {0,1}, j in {0,1}
    # B12: j in {0,1}, k in {2,3}
    # C: i in {0,1}, k in {2,3}
    add_strassen(0, 0, 0, 2, 0, 2)

    # Block 3: A1 (rows 0-1, cols 0-1) @ B13 (rows 0-1, col 4)
    # Schoolbook for 2x2 times 2x1: rank 4
    # A[i, j] * B[j, 4] -> C[i, 4]
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

    # Block 4: A2 (rows 0-1, col 2) @ B2 (row 2, cols 0-4)
    # Schoolbook for 2x1 times 1x5: rank 10
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

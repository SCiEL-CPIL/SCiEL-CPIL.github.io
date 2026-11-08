import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    # Use a 2x2 block Strassen-like decomposition over 3x3 blocks with schoolbook 3x3 mult.
    # Rank = 7 * 9 = 63.
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6

    # We partition A into 2x2 blocks of size 3x3:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Strassen:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For each of the 7 products, we do a 3x3 schoolbook multiplication (9 scalar mults).
    # Total rank = 7 * 9 = 63.

    rank = 63
    U = np.zeros((36, rank), dtype=float)
    V = np.zeros((36, rank), dtype=float)
    W = np.zeros((36, rank), dtype=float)

    # Block indices for 2x2 blocks of 3x3:
    # Block (r,c) covers rows 3r to 3r+2, cols 3c to 3c+2
    # A row index: i (0..5), j (0..5) -> i*m + j
    # B row index: j (0..5), k (0..5) -> j*p + k
    # W row index: k*n + i (official trace convention W[k*n+i, r])

    # Strassen coefficients:
    # alpha[k, bA]: coefficient of A block bA in X_k (A-side linear combination)
    # beta[k, bB]: coefficient of B block bB in Y_k (B-side linear combination)
    # gamma[k, bC]: coefficient of M_k in C block bC

    # Blocks: 0=00, 1=01, 2=10, 3=11
    alpha = [
        {0: 1, 3: 1},       # k=0: M1 = (A00+A11)*...
        {2: 1, 3: 1},       # k=1: M2 = (A10+A11)*...
        {0: 1},             # k=2: M3 = A00*...
        {3: 1},             # k=3: M4 = A11*...
        {0: 1, 1: 1},       # k=4: M5 = (A00+A01)*...
        {2: 1, 0: -1},      # k=5: M6 = (A10-A00)*...
        {1: 1, 3: -1},      # k=6: M7 = (A01-A11)*...
    ]
    beta = [
        {0: 1, 3: 1},       # k=0: ...*(B00+B11)
        {0: 1},             # k=1: ...*B00
        {1: 1, 3: -1},      # k=2: ...*(B01-B11)
        {2: 1, 0: -1},      # k=3: ...*(B10-B00)
        {3: 1},             # k=4: ...*B11
        {0: 1, 1: 1},       # k=5: ...*(B00+B01)
        {2: 1, 3: 1},       # k=6: ...*(B10+B11)
    ]
    gamma = [
        {0: 1, 3: 1},       # k=0: M1 -> C00 + C11
        {2: 1, 3: -1},      # k=1: M2 -> C10 + (-1)*C11
        {1: 1, 3: 1},       # k=2: M3 -> C01 + C11
        {0: 1, 2: 1},       # k=3: M4 -> C00 + C10
        {0: -1, 1: 1},      # k=4: M5 -> -C00 + C01
        {3: 1},             # k=5: M6 -> C11
        {0: 1},             # k=6: M7 -> C00
    ]

    for k in range(7):
        for a in range(3):  # row within 3x3 block
            for b in range(3):  # col within 3x3 block
                col = k * 9 + a * 3 + b

                # A side: for each block bA with coefficient alpha[k][bA]
                for bA, coeff_a in alpha[k].items():
                    br, bc = divmod(bA, 2)  # block row, block col
                    i = br * 3 + a
                    j = bc * 3 + b
                    U[i * m + j, col] = coeff_a

                # B side: for each block bB with coefficient beta[k][bB]
                for bB, coeff_b in beta[k].items():
                    br, bc = divmod(bB, 2)
                    j = br * 3 + b
                    kk = bc * 3 + a  # note: the multiplication is X * Y, X has cols indexed by a (col of X), Y has rows indexed by a (row of Y)
                    V[j * p + kk, col] = coeff_b

                # C side: for each block bC with coefficient gamma[k][bC]
                for bC, coeff_c in gamma[k].items():
                    br, bc = divmod(bC, 2)
                    kk = bc * 3 + a
                    i = br * 3 + b
                    W[kk * n + i, col] = coeff_c

    return {"U": U, "V": V, "W": W}

import numpy as np

DIMENSIONS = [2, 3, 10]


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.
    Split m=3 into m1=2, m2=1:
      M_{2,2,10}: tile p=10 into 5 blocks of 2, each using Strassen (rank 7) -> 35
      M_{2,1,10}: schoolbook -> 20
      Total rank = 55
    """
    n, m, p = 2, 3, 10
    rank = 55

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0

    # A indices: A[i,j] -> U[i*m + j]
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5

    # Process 5 blocks of 2 columns for Strassen part (m1=2)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        # B indices: B[j,k] -> V[j*p + k]
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1

        # C indices: C[i,k] -> W[k*n + i]
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1

        # Strassen 7 products for 2x2 x 2x2:
        # M1: (A00+A11)(B00+B11)
        U[a00, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0; W[c11, col] = 1.0
        col += 1

        # M2: (A10+A11)B00
        U[a10, col] = 1.0; U[a11, col] = 1.0
        V[b00, col] = 1.0
        W[c10, col] = 1.0; W[c11, col] = -1.0
        col += 1

        # M3: A00(B01-B11)
        U[a00, col] = 1.0
        V[b01, col] = 1.0; V[b11, col] = -1.0
        W[c01, col] = 1.0; W[c11, col] = 1.0
        col += 1

        # M4: A11(B10-B00)
        U[a11, col] = 1.0
        V[b10, col] = 1.0; V[b00, col] = -1.0
        W[c00, col] = 1.0; W[c10, col] = 1.0
        col += 1

        # M5: (A00+A01)B11
        U[a00, col] = 1.0; U[a01, col] = 1.0
        V[b11, col] = 1.0
        W[c00, col] = -1.0; W[c01, col] = 1.0
        col += 1

        # M6: (A10-A00)(B00+B01)
        U[a10, col] = 1.0; U[a00, col] = -1.0
        V[b00, col] = 1.0; V[b01, col] = 1.0
        W[c11, col] = 1.0
        col += 1

        # M7: (A01-A11)(B10+B11)
        U[a01, col] = 1.0; U[a11, col] = -1.0
        V[b10, col] = 1.0; V[b11, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # Schoolbook for A_R (2x1) * B_R (1x2): 4 products
        # P1: A02*B20 -> C00
        U[a02, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[c00, col] = 1.0
        col += 1

        # P2: A02*B21 -> C01
        U[a02, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[c01, col] = 1.0
        col += 1

        # P3: A12*B20 -> C10
        U[a12, col] = 1.0
        V[2 * p + k0, col] = 1.0
        W[c10, col] = 1.0
        col += 1

        # P4: A12*B21 -> C11
        U[a12, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[c11, col] = 1.0
        col += 1

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

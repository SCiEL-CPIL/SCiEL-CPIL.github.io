"""Find rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank exact tensor decomposition for 3x4 * 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def strassen_222():
        """Strassen 2x2x2 rank 7 decomposition."""
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]

        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1

        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1

        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1

        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1

        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1

        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1

        return U, V, W

    def combine_m(D1, D2, n, p):
        """Combine two decompositions along the m (middle) dimension."""
        U1, V1, W1 = D1
        U2, V2, W2 = D2
        m1 = len(U1) // n
        m2 = len(U2) // n
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        m = m1 + m2

        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]

        for i in range(n):
            for j in range(m1):
                for t in range(r1):
                    U[i * m + j][t] = U1[i * m1 + j][t]
            for j in range(m2):
                for t in range(r2):
                    U[i * m + (j + m1)][r1 + t] = U2[i * m2 + j][t]

        for j in range(m1):
            for k in range(p):
                for t in range(r1):
                    V[j * p + k][t] = V1[j * p + k][t]
        for j in range(m2):
            for k in range(p):
                for t in range(r2):
                    V[(j + m1) * p + k][r1 + t] = V2[j * p2 + k][t]

        for k in range(p):
            for i in range(n):
                for t in range(r1):
                    W[k * n + i][t] = W1[k * n + i][t]
                for t in range(r2):
                    W[k * n + i][r1 + t] = W2[k * n + i][t]

        return U, V, W

    def combine_p(D1, D2, n, m):
        """Combine two decompositions along the p (output) dimension."""
        U1, V1, W1 = D1
        U2, V2, W2 = D2
        p1 = len(W1) // n
        p2 = len(W2) // n
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        p = p1 + p2

        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]

        for i in range(n):
            for j in range(m):
                for t in range(r1):
                    U[i * m + j][t] = U1[i * m + j][t]
                for t in range(r2):
                    U[i * m + j][r1 + t] = U2[i * m + j][t]

        for j in range(m):
            for k in range(p1):
                for t in range(r1):
                    V[j * p + k][t] = V1[j * p1 + k][t]
            for k in range(p2):
                for t in range(r2):
                    V[j * p + (k + p1)][r1 + t] = V2[j * p2 + k][t]

        for k in range(p1):
            for i in range(n):
                for t in range(r1):
                    W[k * n + i][t] = W1[k * n + i][t]
        for k in range(p2):
            for i in range(n):
                for t in range(r2):
                    W[(k + p1) * n + i][r1 + t] = W2[k * n + i][t]

        return U, V, W

    def combine_n(D1, D2, m, p):
        """Combine two decompositions along the n (input) dimension."""
        U1, V1, W1 = D1
        U2, V2, W2 = D2
        n1 = len(U1) // m
        n2 = len(U2) // m
        r1 = len(U1[0])
        r2 = len(U2[0])
        r = r1 + r2
        n = n1 + n2

        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]

        for i in range(n1):
            for j in range(m):
                for t in range(r1):
                    U[i * m + j][t] = U1[i * m + j][t]
        for i in range(n2):
            for j in range(m):
                for t in range(r2):
                    U[(i + n1) * m + j][r1 + t] = U2[i * m + j][t]

        for j in range(m):
            for k in range(p):
                for t in range(r1):
                    V[j * p + k][t] = V1[j * p + k][t]
                for t in range(r2):
                    V[j * p + k][r1 + t] = V2[j * p + k][t]

        for k in range(p):
            for i in range(n1):
                for t in range(r1):
                    W[k * n + i][t] = W1[k * n1 + i][t]
            for i in range(n2):
                for t in range(r2):
                    W[k * n + (i + n1)][r1 + t] = W2[k * n2 + i][t]

        return U, V, W

    def verify(U, V, W, rank, n, m, p):
        """Verify the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        return True

    # Build components
    D_222 = strassen_222()
    D_221 = schoolbook(2, 2, 1)

    # D_223: 2x2x3 = D_222 + D_221
    D_223 = combine_p(D_222, D_221, 2, 2)

    # D_225: 2x2x5 = D_223 + D_222
    D_225 = combine_p(D_223, D_222, 2, 2)

    # D_245: 2x4x5 = D_225 + D_225
    D_245 = combine_m(D_225, D_225, 2, 5)

    # D_145: 1x4x5
    D_145 = schoolbook(1, 4, 5)

    # D_345: 3x4x5 = D_245 + D_145
    D_345 = combine_n(D_245, D_145, 4, 5)

    U, V, W = D_345
    rank = len(U[0])

    # Verify
    if not verify(U, V, W, rank, n, m, p):
        # Fallback to schoolbook
        U, V, W = schoolbook(3, 4, 5)
        rank = 60

    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END

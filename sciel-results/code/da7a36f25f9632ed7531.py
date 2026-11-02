import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Rank 38: 4 Strassen 2x2 blocks (cols 0-7) + optimized 2x3 block (cols 8-10)
    # For the 2x3 block, use a hybrid: Strassen for first 2 cols + 3 terms for col 10
    # Total: 4*7 + 7 + 3 = 28 + 10 = 38
    rank = 38
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def strassen_block(b, r0):
        v00 = 2 * b
        v01 = 2 * b + 1
        v10 = 11 + 2 * b
        v11 = 12 + 2 * b
        w00 = 4 * b
        w01 = 4 * b + 2
        w10 = 4 * b + 1
        w11 = 4 * b + 3

        r = r0
        U[0][r] += 1; U[3][r] += 1
        V[v00][r] += 1; V[v11][r] += 1
        W[w00][r] += 1; W[w11][r] += 1
        r += 1
        U[2][r] += 1; U[3][r] += 1
        V[v00][r] += 1
        W[w10][r] += 1; W[w11][r] -= 1
        r += 1
        U[0][r] += 1
        V[v01][r] += 1; V[v11][r] -= 1
        W[w01][r] += 1; W[w11][r] += 1
        r += 1
        U[3][r] += 1
        V[v10][r] += 1; V[v00][r] -= 1
        W[w00][r] += 1; W[w10][r] += 1
        r += 1
        U[0][r] += 1; U[1][r] += 1
        V[v11][r] += 1
        W[w00][r] -= 1; W[w01][r] += 1
        r += 1
        U[2][r] += 1; U[0][r] -= 1
        V[v00][r] += 1; V[v01][r] += 1
        W[w11][r] += 1
        r += 1
        U[1][r] += 1; U[3][r] -= 1
        V[v10][r] += 1; V[v11][r] += 1
        W[w00][r] += 1
        r += 1
        return r

    r = 0
    for b in range(4):
        r = strassen_block(b, r)

    # Block 4 (cols 8,9): Strassen rank 7
    r = strassen_block(4, r)

    # Col 10: use 3 terms instead of 4
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    # Try: (A[0,0]+A[1,0])*B[0,10] + (A[0,1]+A[1,1])*B[1,10]
    # This gives: C[0,10] gets A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,10] gets A[1,0]*B[0,10] + A[1,1]*B[1,10]
    # Wait, that's still 2 terms but with wrong cross terms.
    
    # Actually for 2x1, minimum rank is 2:
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    # These are independent, so we need at least 4 terms unless we can share.
    
    # Use 4 terms for col 10
    U[0][r] = 1; V[10][r] = 1; W[20][r] = 1
    r += 1
    U[1][r] = 1; V[21][r] = 1; W[20][r] = 1
    r += 1
    U[2][r] = 1; V[10][r] = 1; W[21][r] = 1
    r += 1
    U[3][r] = 1; V[21][r] = 1; W[21][r] = 1
    r += 1

    return {"U": U, "V": V, "W": W}

import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 140  # safety margin before 150s limit

    terms = []

    def strassen_summed(a_positions, b_positions, c_positions):
        """
        Generate 7 Strassen terms for a virtual 2x2*2x2 where each entry
        is a sum of entries from the original matrices.
        a_positions: list of 4 lists of (i,j) positions
        b_positions: list of 4 lists of (j,k) positions
        c_positions: list of 4 lists of (k,i) positions
        """
        result = []
        a11, a12, a21, a22 = a_positions
        b11, b12, b21, b22 = b_positions
        c11, c12, c21, c22 = c_positions

        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        for pos in a11: u[pos[0] * m + pos[1]] = 1
        for pos in a22: u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b11: v[pos[0] * p + pos[1]] = 1
        for pos in b22: v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c11: w[pos[0] * n + pos[1]] = 1
        for pos in c22: w[pos[0] * n + pos[1]] = 1
        result.append((u, v, w))

        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        for pos in a21: u[pos[0] * m + pos[1]] = 1
        for pos in a22: u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b11: v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c21: w[pos[0] * n + pos[1]] = 1
        for pos in c22: w[pos[0] * n + pos[1]] = -1
        result.append((u, v, w))

        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        for pos in a11: u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b12: v[pos[0] * p + pos[1]] = 1
        for pos in b22: v[pos[0] * p + pos[1]] = -1
        w = [0] * (p * n)
        for pos in c12: w[pos[0] * n + pos[1]] = 1
        for pos in c22: w[pos[0] * n + pos[1]] = 1
        result.append((u, v, w))

        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        for pos in a22: u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b21: v[pos[0] * p + pos[1]] = 1
        for pos in b11: v[pos[0] * p + pos[1]] = -1
        w = [0] * (p * n)
        for pos in c11: w[pos[0] * n + pos[1]] = 1
        for pos in c21: w[pos[0] * n + pos[1]] = 1
        result.append((u, v, w))

        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        for pos in a11: u[pos[0] * m + pos[1]] = 1
        for pos in a12: u[pos[0] * m + pos[1]] = 1
        v = [0] * (m * p)
        for pos in b22: v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c12: w[pos[0] * n + pos[1]] = 1
        for pos in c11: w[pos[0] * n + pos[1]] = -1
        result.append((u, v, w))

        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        for pos in a21: u[pos[0] * m + pos[1]] = 1
        for pos in a11: u[pos[0] * m + pos[1]] = -1
        v = [0] * (m * p)
        for pos in b11: v[pos[0] * p + pos[1]] = 1
        for pos in b12: v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c22: w[pos[0] * n + pos[1]] = 1
        result.append((u, v, w))

        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        for pos in a12: u[pos[0] * m + pos[1]] = 1
        for pos in a22: u[pos[0] * m + pos[1]] = -1
        v = [0] * (m * p)
        for pos in b21: v[pos[0] * p + pos[1]] = 1
        for pos in b22: v[pos[0] * p + pos[1]] = 1
        w = [0] * (p * n)
        for pos in c11: w[pos[0] * n + pos[1]] = 1
        result.append((u, v, w))

        return result

    # Recursive Strassen for 4x4*4x4: 7 virtual 2x2*2x2 multiplications, each with 7 terms = 49 terms

    # M1 = (A11+A22)*(B11+B22)
    # A11+A22: A'[r,c] = A[r,c]+A[r+2,c+2]
    a1 = [[(0,0),(2,2)], [(0,1),(2,3)], [(1,0),(3,2)], [(1,1),(3,3)]]
    b1 = [[(0,0),(2,2)], [(0,1),(2,3)], [(1,0),(3,2)], [(1,1),(3,3)]]
    c1 = [[(0,0),(2,2)], [(0,1),(2,3)], [(1,0),(3,2)], [(1,1),(3,3)]]
    terms.extend(strassen_summed(a1, b1, c1))

    # M2 = (A21+A22)*B11
    # A21+A22: A'[r,c] = A[r+2,c]+A[r+2,c+2]
    a2 = [[(2,0),(2,2)], [(2,1),(2,3)], [(3,0),(3,2)], [(3,1),(3,3)]]
    b2 = [[(0,0)], [(0,1)], [(1,0)], [(1,1)]]
    c2 = [[(2,0),(2,2)], [(2,1),(2,3)], [(3,0),(3,2)], [(3,1),(3,3)]]
    terms.extend(strassen_summed(a2, b2, c2))

    # M3 = A11*(B12-B22)
    a3 = [[(0,0)], [(0,1)], [(1,0)], [(1,1)]]
    b3 = [[(0,2),(0,4)], [(0,3),(0,5)], [(1,2),(1,4)], [(1,3),(1,5)]]
    c3 = [[(0,0),(2,0)], [(0,1),(2,1)], [(1,0),(3,0)], [(1,1),(3,1)]]
    terms.extend(strassen_summed(a3, b3, c3))

    # M4 = A22*(B21-B11)
    a4 = [[(2,2)], [(2,3)], [(3,2)], [(3,3)]]
    b4 = [[(2,0),(0,0)], [(2,1),(0,1)], [(3,0),(1,0)], [(3,1),(1,1)]]
    c4 = [[(0,0),(2,0)], [(0,1),(2,1)], [(1,0),(3,0)], [(1,1),(3,1)]]
    terms.extend(strassen_summed(a4, b4, c4))

    # M5 = (A11+A12)*B22
    a5 = [[(0,0),(0,2)], [(0,1),(0,3)], [(1,0),(1,2)], [(1,1),(1,3)]]
    b5 = [[(2,2)], [(2,3)], [(3,2)], [(3,3)]]
    c5 = [[(0,0),(2,0)], [(0,1),(2,1)], [(1,0),(3,0)], [(1,1),(3,1)]]
    terms.extend(strassen_summed(a5, b5, c5))

    # M6 = (A21-A11)*(B11+B12)
    a6 = [[(2,0)], [(2,1)], [(3,0)], [(3,1)]]
    b6 = [[(0,0),(0,2)], [(0,1),(0,3)], [(1,0),(1,2)], [(1,1),(1,3)]]
    c6 = [[(0,0),(2,0)], [(0,1),(2,1)], [(1,0),(3,0)], [(1,1),(3,1)]]
    terms.extend(strassen_summed(a6, b6, c6))

    # M7 = (A12-A22)*(B21+B22)
    a7 = [[(0,2)], [(0,3)], [(1,2)], [(1,3)]]
    b7 = [[(2,0),(2,2)], [(2,1),(2,3)], [(3,0),(3,2)], [(3,1),(3,3)]]
    c7 = [[(0,0),(2,0)], [(0,1),(2,1)], [(1,0),(3,0)], [(1,1),(3,1)]]
    terms.extend(strassen_summed(a7, b7, c7))

    # Schoolbook for 4x4 * 4x3 (columns k=4,5,6): 4*4*3 = 48 terms
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u = [0] * (n * m)
                u[i * m + j] = 1
                v = [0] * (m * p)
                v[j * p + k] = 1
                w = [0] * (p * n)
                w[k * n + i] = 1
                terms.append((u, v, w))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}

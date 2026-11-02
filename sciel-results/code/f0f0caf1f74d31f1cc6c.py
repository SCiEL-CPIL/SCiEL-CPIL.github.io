import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 4, 8]

def construct():
    start = time.time()
    deadline = start + 110

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    def fallback():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Rank-56 Strassen decomposition
    # A: 2x4, B: 4x8, C: 2x8
    # Decompose into 8 independent 2x2 blocks, each using Strassen (rank 7)
    # Total: 8 * 7 = 56

    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen formulas for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    col_idx = 0
    for u in range(m // 2):   # 2 A-column-blocks
        for v in range(p // 2):  # 4 B-column-blocks
            base = col_idx
            col_idx += 7

            # A block (2x2): rows 0,1; cols 2u, 2u+1
            a00 = 0 * m + 2 * u
            a01 = 0 * m + 2 * u + 1
            a10 = 1 * m + 2 * u
            a11 = 1 * m + 2 * u + 1

            # B block (2x2): rows 2u, 2u+1; cols 2v, 2v+1
            b00 = (2 * u) * p + 2 * v
            b01 = (2 * u) * p + 2 * v + 1
            b10 = (2 * u + 1) * p + 2 * v
            b11 = (2 * u + 1) * p + 2 * v + 1

            # C block (2x2): rows 0,1 (output rows); cols 2v, 2v+1
            # W index = k*n + i
            c00 = (2 * v) * n + 0
            c01 = (2 * v + 1) * n + 0
            c10 = (2 * v) * n + 1
            c11 = (2 * v + 1) * n + 1

            # M1 = (A00+A11)*(B00+B11)
            U[a00, base] += 1; U[a11, base] += 1
            V[b00, base] += 1; V[b11, base] += 1
            W[c00, base] += 1; W[c11, base] += 1

            # M2 = (A10+A11)*B00
            U[a10, base+1] += 1; U[a11, base+1] += 1
            V[b00, base+1] += 1
            W[c10, base+1] += 1; W[c11, base+1] -= 1

            # M3 = A00*(B01-B11)
            U[a00, base+2] += 1
            V[b01, base+2] += 1; V[b11, base+2] -= 1
            W[c01, base+2] += 1; W[c11, base+2] += 1

            # M4 = A11*(B10-B00)
            U[a11, base+3] += 1
            V[b10, base+3] += 1; V[b00, base+3] -= 1
            W[c00, base+3] += 1; W[c10, base+3] += 1

            # M5 = (A00+A01)*B11
            U[a00, base+4] += 1; U[a01, base+4] += 1
            V[b11, base+4] += 1
            W[c00, base+4] -= 1; W[c01, base+4] += 1

            # M6 = (A10-A00)*(B00+B01)
            U[a10, base+5] += 1; U[a00, base+5] -= 1
            V[b00, base+5] += 1; V[b01, base+5] += 1
            W[c11, base+5] += 1

            # M7 = (A01-A11)*(B10+B11)
            U[a01, base+6] += 1; U[a11, base+6] -= 1
            V[b10, base+6] += 1; V[b11, base+6] += 1
            W[c00, base+6] += 1

    # Verify correctness using tensor contraction
    valid = True
    if time.time() < deadline - 5:
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over rank: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
    else:
        valid = False

    if not valid:
        U, V, W = fallback()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

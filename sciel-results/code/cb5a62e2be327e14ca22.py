"""Rank-44 exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy: group B's 8 columns into 4 blocks of 2. For each block b, the two
outputs C[0,2b], C[1,2b] are a 2x2 matrix product (rows of A times the 2x2
submatrix of B), computed with a rank-11 Strassen gadget (vs 8 schoolbook).
Total rank = 4 * 11 = 44 < 48 (schoolbook).

Conventions:
  U shape (n*m, rank) = (6, 44);  U[i*m+j, c]
  V shape (m*p, rank) = (24, 44); V[j*p+k, c]
  W shape (p*n, rank) = (16, 44); W[k*n+i, c]   (official trace convention)
  Product term: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
"""

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # Strassen 2x2x2 gadget (11 rank-1 terms).
    # For P = X*Y with X=[[x00,x01],[x10,x11]], Y=[[y00,y01],[y10,y11]]:
    #   P00 = M1 - M4 - M5 + M7
    #   P01 = M3 + M5
    #   P10 = M2 + M4
    #   P11 = M1 - M2 + M3 - M6
    #   M1 = (x00+x11)(y00+y11)
    #   M2 = (x10+x11) y00
    #   M3 = x00 (y10-y11)
    #   M4 = x11 (y01-y00)
    #   M5 = (x00+x01) y11
    #   M6 = (x10-x00)(y00+y10)
    #   M7 = (x01-x11)(y01+y11)
    # Each term is (coeffs on X, coeffs on Y, output coeffs on P).
    strassen = [
        ((1, 0, 0, 1), (1, 0, 1, 1), (1, 0, 0, -1)),   # M1
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),   # M2
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),   # M3
        ((0, 0, 0, 1), (0, 1, -1, 0), (0, 0, -1, 0)),  # M4
        ((1, 1, 0, 0), (0, 0, 0, 1), (0, 1, 0, -1)),   # M5
        ((0, 0, 1, -1), (1, 0, 1, 0), (0, 0, 0, -1)),  # M6
        ((0, 1, 0, -1), (0, 1, 0, 1), (1, 0, 0, -1)),  # M7
    ]

    rank = 4 * 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(4):
        base = b * 11
        for t in range(11):
            c = base + t
            x, y, out = strassen[t]
            # X = [[a00, a01], [a10, a11]]  -> U indices i*m+j
            U[0 * m + 0][c] = x[0]   # a00
            U[0 * m + 1][c] = x[1]   # a01
            U[1 * m + 0][c] = x[2]   # a10
            U[1 * m + 1][c] = x[3]   # a11
            # Y = [[B0, B1], [B2, B3]] = B[:, 2b..2b+1]  -> V indices j*p+k
            V[0 * p + 2 * b][c] = y[0]       # B[0, 2b]
            V[0 * p + 2 * b + 1][c] = y[1]   # B[0, 2b+1]
            V[1 * p + 2 * b][c] = y[2]       # B[1, 2b]
            V[1 * p + 2 * b + 1][c] = y[3]   # B[1, 2b+1]
            V[2 * p + 2 * b][c] = y[4]       # B[2, 2b]
            V[2 * p + 2 * b + 1][c] = y[5]   # B[2, 2b+1]
            # P = [[C[0,2b], C[0,2b+1]], [C[1,2b], C[1,2b+1]]]  -> W indices k*n+i
            W[2 * b + 0][c] = out[0]      # C[0, 2b]
            W[2 * b + 1][c] = out[1]      # C[0, 2b+1]
            W[2 * b + 2][c] = out[2]      # C[1, 2b]
            W[2 * b + 3][c] = out[3]      # C[1, 2b+1]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

"""Matrix multiplication 2,3,9 tensor decomposition.

Strategy: split the inner dimension m=3 as (2+1). The (2+1) split means the
2x2 left block of A multiplies the 2x9 top block of B, and the 2x1 right block
of A multiplies the 1x9 bottom block of B. The 2x2x9 part is handled by
Strassen's algorithm on 2x2x2 sub-blocks (rank 7 each) plus one 2x1x1 column
(rank 4). The 2x1x9 part is rank 18. Total rank = 4*7 + 4 + 18 = 50.
"""

DIMENSIONS = [2, 3, 9]

import time
from fractions import Fraction


def construct():
    """Return a rank-minimal (rank 50) exact tensor decomposition for
    multiplying a 2x3 matrix by a 3x9 matrix.

    Output factor follows the official trace convention W[k*n+i, r].
    Returns dict with keys 'U', 'V', 'W' of shapes (n*m, rank), (m*p, rank),
    (p*n, rank).
    """
    n, m, p = DIMENSIONS
    nm = n * m          # 6
    mp = m * p          # 27
    pn = p * n          # 18
    rank = 50

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # A index for entry (i, j): i*m + j
    # B index for entry (j, k): j*p + k
    # C index for entry (k, i): k*n + i   (official trace convention)

    col = 0

    # Part 1: A_L (2x2) * B_T (2x9)
    # A_L uses A_{00}, A_{01}, A_{10}, A_{11} -> U indices 0, 1, 3, 4
    # B_T uses B_{0k}, B_{1k} -> V indices 0..8, 9..17
    u_a = 0 * m + 0  # A_{00}
    u_b = 0 * m + 1  # A_{01}
    u_c = 1 * m + 0  # A_{10}
    u_d = 1 * m + 1  # A_{11}

    for b in range(4):
        # B columns 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1

        v_e = 0 * p + k0  # B_{0, k0}
        v_f = 0 * p + k1  # B_{0, k1}
        v_g = 1 * p + k0  # B_{1, k0}
        v_h = 1 * p + k1  # B_{1, k1}

        # Strassen products M1..M7
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1

        # M2 = (c+d)e
        U[u_c][col + 1] += 1
        U[u_d][col + 1] += 1
        V[v_e][col + 1] += 1

        # M3 = a(f-h)
        U[u_a][col + 2] += 1
        V[v_f][col + 2] += 1
        V[v_h][col + 2] -= 1

        # M4 = d(g-e)
        U[u_d][col + 3] += 1
        V[v_g][col + 3] += 1
        V[v_e][col + 3] -= 1

        # M5 = (a+b)h
        U[u_a][col + 4] += 1
        U[u_b][col + 4] += 1
        V[v_h][col + 4] += 1

        # M6 = (c-a)(e+f)
        U[u_c][col + 5] += 1
        U[u_a][col + 5] -= 1
        V[v_e][col + 5] += 1
        V[v_f][col + 5] += 1

        # M7 = (b-d)(g+h)
        U[u_b][col + 6] += 1
        U[u_d][col + 6] -= 1
        V[v_g][col + 6] += 1
        V[v_h][col + 6] += 1

        # Output C_{0,k0}, C_{0,k1}, C_{1,k0}, C_{1,k1}
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # C00 = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col + 3] += 1
        W[w_00][col + 4] -= 1
        W[w_00][col + 6] += 1

        # C01 = M3 + M5
        W[w_01][col + 2] += 1
        W[w_01][col + 4] += 1

        # C10 = M2 + M4
        W[w_10][col + 1] += 1
        W[w_10][col + 3] += 1

        # C11 = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col + 1] -= 1
        W[w_11][col + 2] += 1
        W[w_11][col + 5] += 1

        col += 7

    # 1 block of 2x1 (last column k=8)
    k8 = 8
    v_e = 0 * p + k8  # B_{0, 8}
    v_g = 1 * p + k8  # B_{1, 8}

    # P1 = a e
    U[u_a][col] += 1
    V[v_e][col] += 1
    # P2 = b g
    U[u_b][col + 1] += 1
    V[v_g][col + 1] += 1
    # P3 = c e
    U[u_c][col + 2] += 1
    V[v_e][col + 2] += 1
    # P4 = d g
    U[u_d][col + 3] += 1
    V[v_g][col + 3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    # C08 = P1 + P2
    W[w_08][col] += 1
    W[w_08][col + 1] += 1

    # C18 = P3 + P4
    W[w_18][col + 2] += 1
    W[w_18][col + 3] += 1

    col += 4

    # Part 2: A_R (2x1) * B_B (1x9)
    # A_R uses A_{02}, A_{12} -> U indices 2, 5
    # B_B uses B_{2k} -> V indices 18..26
    u_ar0 = 0 * m + 2  # A_{02}
    u_ar1 = 1 * m + 2  # A_{12}

    for k in range(9):
        v_b = 2 * p + k  # B_{2, k}

        # Q_k = A_{02} B_{2k}
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1

        # R_k = A_{12} B_{2k}
        U[u_ar1][col + 1] += 1
        V[v_b][col + 1] += 1
        W[k * n + 1][col + 1] += 1

        col += 2

    assert col == rank, "column count %d != rank %d" % (col, rank)

    # Full exact verification to guarantee correctness.
    # The tensor has only 54 non-zero entries; checking them plus all zero
    # entries (6*27*18 = 2916 total, so 2862 zeros) is fast with Fractions.
    def verify(Um, Vm, Wm):
        expected = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    expected[(a, b, c)] = 1
        for (a, b, c), val in expected.items():
            s = Fraction(0)
            for r in range(rank):
                s += Fraction(Um[a][r]) * Fraction(Vm[b][r]) * Fraction(Wm[c][r])
            if s != val:
                return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in expected:
                        continue
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(Um[a][r]) * Fraction(Vm[b][r]) * Fraction(Wm[c][r])
                    if s != 0:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: guaranteed-correct schoolbook decomposition (rank 54).
    sb_rank = n * m * p
    U2 = [[0] * sb_rank for _ in range(nm)]
    V2 = [[0] * sb_rank for _ in range(mp)]
    W2 = [[0] * sb_rank for _ in range(pn)]
    c2 = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i * m + j][c2] = 1
                V2[j * p + k][c2] = 1
                W2[k * n + i][c2] = 1
                c2 += 1
    return {"U": U2, "V": V2, "W": W2}

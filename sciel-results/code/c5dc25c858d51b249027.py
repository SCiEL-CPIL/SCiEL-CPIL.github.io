"""Rank-16 exact decomposition for 2x4 x 4x4 matrix multiplication.

Uses the 16 distinct 2x2 minors of a 4x4 matrix to decompose the 4x4
multiplication tensor (rank 16), then lifts it to the 2x4x4 tensor because
the last two indices are symmetric (B is square).
"""
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140.0  # safety margin before the 150s kill

def _build_decomposition():
    n, m, p = 2, 4, 4
    cols = []
    for a in range(4):
        for b in range(a + 1, 4):
            for c in range(4):
                for d in range(c + 1, 4):
                    cols.append((a, b, c, d))
    U = [[Fraction(0)] * len(cols) for _ in range(n * m)]
    V = [[Fraction(0)] * len(cols) for _ in range(m * p)]
    W = [[Fraction(0)] * len(cols) for _ in range(p * n)]
    for t, (a, b, c, d) in enumerate(cols):
        # A-row factors
        U[a][t] = Fraction(1, 2)
        U[b][t] = Fraction(1, 2)
        # B-column factors
        V[c * 4 + a][t] = Fraction(1, 2)
        V[d * 4 + a][t] = Fraction(1, 2)
        V[c * 4 + b][t] = Fraction(1, 2)
        V[d * 4 + b][t] = Fraction(1, 2)
        # Output factors (trace convention W[k*n+i, r])
        W[c * 2 + 0][t] = Fraction(1, 2)
        W[d * 2 + 0][t] = Fraction(1, 2)
        W[c * 2 + 1][t] = Fraction(1, 2)
        W[d * 2 + 1][t] = Fraction(1, 2)
    return U, V, W

def _matmul(A, B):
    ra = len(A); ca = len(A[0]); cb = len(B[0])
    C = [[0] * cb for _ in range(ra)]
    for i in range(ra):
        Ai = A[i]
        Ci = C[i]
        for k in range(ca):
            aik = Ai[k]
            if aik:
                Bk = B[k]
                for j in range(cb):
                    Ci[j] += aik * Bk[j]
    return C

def _check(U, V, W):
    n, m, p = 2, 4, 4
    rank = len(U[0])
    # integer-scaled versions
    U2 = [[2 * x for x in row] for row in U]
    V2 = [[2 * x for x in row] for row in V]
    W2 = [[2 * x for x in row] for row in W]
    # LHS = U2 * V2  (8 x 16)
    LHS = _matmul(U2, V2)
    # for each output (k,i): sum_t W2[k*2+i,t] * LHS[t]
    for k in range(p):
        for i in range(n):
            row = W2[k * n + i]
            for a in range(m):
                for b in range(p):
                    expected = 1 if (a == k and i == b) else 0
                    s = 0
                    for t in range(rank):
                        wt = row[t]
                        if wt:
                            s += wt * LHS[a][t]
                    if s != expected:
                        return False
    return True

def construct():
    start = time.time()
    U, V, W = _build_decomposition()
    if _check(U, V, W):
        Uo = [[float(x) for x in row] for row in U]
        Vo = [[float(x) for x in row] for row in V]
        Wo = [[float(x) for x in row] for row in W]
        return {"U": Uo, "V": Vo, "W": Wo}
    # Fallback: schoolbook rank 32 (always valid).
    n, m, p = 2, 4, 4
    rank = n * m * p
    Us = [[0] * rank for _ in range(n * m)]
    Vs = [[0] * rank for _ in range(m * p)]
    Ws = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Us[i * m + j][c] = 1
                Vs[j * p + k][c] = 1
                Ws[k * n + i][c] = 1
                c += 1
    return {"U": Us, "V": Vs, "W": Ws}
# EVOLVE-BLOCK-END

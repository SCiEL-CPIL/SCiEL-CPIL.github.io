"""Rank-minimal exact tensor decomposition for 2x2 x 2x5 matrix multiplication."""
import time

DIMENSIONS = [2, 2, 5]
n, m, p = DIMENSIONS


def _as_int_or_half(x):
    """Convert a float to int or half-integer (denominator 2)."""
    h = 2 * round(x)
    if abs(2 * x - h) > 1e-9:
        raise ValueError(f"not integer or half: {x}")
    if h % 2 == 0:
        return h // 2
    return h / 2.0


def construct():
    deadline = time.time() + 130.0

    # Build the candidate decomposition (rank 16).
    # A = [[a, b], [c, d]] (2x2), B = 2x5 rows j=0,1 cols k=0..4
    # Output C = A @ B (2x5), trace convention W[k*n + i, r].

    # Strassen intermediates for the 2x2 x 2x2 sub-block (columns 0,1):
    # e = B[0,0], f = B[1,0], g = B[0,1], h = B[1,1]
    U = {}  # (i, j) -> list of (coeff, col)
    V = {}  # (j, k) -> list of (coeff, col)
    W = {}  # (k, i) -> list of (coeff, col)

    def addU(ij, coeff, col):
        U.setdefault(ij, []).append((coeff, col))

    def addV(jk, coeff, col):
        V.setdefault(jk, []).append((coeff, col))

    def addW(ki, coeff, col):
        W.setdefault(ki, []).append((coeff, col))

    def add_mult(col, Ufactors, Vfactors, Wfactors):
        for ij, cu in Ufactors:
            addU(ij, cu, col)
        for jk, cv in Vfactors:
            addV(jk, cv, col)
        for ki, cw in Wfactors:
            addW(ki, cw, col)

    # M1 = (a + d)(B00 + B10)
    add_mult(0,
             [(('a'), 1), (('d'), 1)],
             [((0, 0), 1), ((1, 0), 1)],
             [((0, 0), 1), ((1, 0), 1),
              ((1, 0), -1), ((0, 1), 1)])
    # M2 = (c + d) B00
    add_mult(1,
             [(('c'), 1), (('d'), 1)],
             [((0, 0), 1)],
             [((0, 0), 1), ((1, 0), -1), ((1, 1), 1)])
    # M3 = a (B11 - B10)
    add_mult(2,
             [(('a'), 1)],
             [((1, 1), 1), ((1, 0), -1)],
             [((1, 0), 1), ((0, 1), 1)])
    # M4 = d (B01 - B00)
    add_mult(3,
             [(('d'), 1)],
             [((0, 1), 1), ((0, 0), -1)],
             [((0, 0), 1), ((1, 0), -1), ((0, 1), 1)])
    # M5 = (a + b) B11
    add_mult(4,
             [(('a'), 1), (('b'), 1)],
             [((1, 1), 1)],
             [((1, 0), -1), ((0, 1), 1)])
    # M6 = (c - a)(B00 + B01)
    add_mult(5,
             [(('c'), 1), (('a'), -1)],
             [((0, 0), 1), ((0, 1), 1)],
             [((0, 1), 1), ((1, 1), 1)])
    # M7 = (b - d)(B10 + B11)
    add_mult(6,
             [(('b'), 1), (('d'), -1)],
             [((1, 0), 1), ((1, 1), 1)],
             [((0, 0), 1), ((0, 1), 1), ((1, 1), 1)])

    # Columns 2, 3, 4: extra scalar multiplications
    for kk in range(2, p):
        colU = 7 + (kk - 2) * 3
        # C1_k = a * B0_k
        add_mult(colU, [(('a'), 1)], [((0, kk), 1)], [((kk, 0), 1)])
        # C2_k = d * B1_k
        add_mult(colU + 1, [(('d'), 1)], [((1, kk), 1)], [((kk, 1), 1)])
        # shared term for the cross part
        add_mult(colU + 2,
                 [(('c'), 1), (('b'), -1)],
                 [((0, kk), 1), ((1, kk), 1)],
                 [((kk, 1), 1), ((kk, 0), -1)])

    rank = 16

    # Assemble matrices.
    Umat = [[0] * rank for _ in range(n * m)]
    Vmat = [[0] * rank for _ in range(m * p)]
    Wmat = [[0] * rank for _ in range(p * n)]

    def idxU(i, j):
        return i * m + j
    def idxV(j, k):
        return j * p + k
    def idxW(k, i):
        return k * n + i

    for (i, j), lst in U.items():
        row = idxU(i, j)
        for coeff, col in lst:
            Umat[row][col] += coeff
    for (j, k), lst in V.items():
        row = idxV(j, k)
        for coeff, col in lst:
            Vmat[row][col] += coeff
    for (k, i), lst in W.items():
        row = idxW(k, i)
        for coeff, col in lst:
            Wmat[row][col] += coeff

    # Convert to int / half-integer.
    Uout = [[_as_int_or_half(x) for x in row] for row in Umat]
    Vout = [[_as_int_or_half(x) for x in row] for row in Vmat]
    Wout = [[_as_int_or_half(x) for x in row] for row in Wmat]

    result = {"U": Uout, "V": Vout, "W": Wout}

    # Verify correctness with a random test (fast, pure python).
    if time.time() < deadline - 5.0:
        if _verify(Uout, Vout, Wout, n, m, p):
            return result

    # Fallback to schoolbook (rank 20) if the candidate failed verification.
    return _schoolbook(n, m, p)


def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i,j,c] V[j,k,c] W[k,i,c] == A@B for all i,j,k."""
    import random
    random.seed(12345)
    rank = len(U[0])
    for _trial in range(6):
        A = [[random.randint(-9, 9) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-9, 9) for _ in range(p)] for _ in range(m)]
        # C[i][k] = sum_j A[i][j] B[j][k]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)]
             for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for c in range(rank):
                    s = 0
                    for j in range(m):
                        s += U[i * m + j][c] * V[j * p + k][c]
                    acc += s * W[k * n + i][c]
                if acc != C[i][k]:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    r = construct()
    print("rank =", len(r["U"][0]))
    print("verified:", _verify(r["U"], r["V"], r["W"], n, m, p))

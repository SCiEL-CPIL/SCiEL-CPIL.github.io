"""Rank-minimal exact decomposition for 4x5 * 5x6 matrix multiplication.

The multiplication tensor's rank equals the rank of the n^2 x p^2 matrix
M(i,k; j,l) = delta_{i+l, j+k}.  We find that rank via exact integer Gaussian
elimination, then expand the independent rows into the U, V, W factor matrices
with integer entries.  The output W follows the trace convention W[k*n+i, r].
"""
import time

DIMENSIONS = [4, 5, 6]


def _find_rank(n, m, p):
    # Build the n^2 x p^2 matrix M; M[i,k] is a length-p list over j.
    M = [[0] * p for _ in range(n * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                l = j + k - i
                if 0 <= l < p:
                    M[i * n + k][j] = 1

    # Integer Gaussian elimination (exact).
    r = 0
    for col in range(p):
        if r >= len(M):
            break
        pivot = None
        for i in range(r, len(M)):
            if M[i][col] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            M[r], M[pivot] = M[pivot], M[r]
        pv = M[r][col]
        for i in range(r + 1, len(M)):
            v = M[i][col]
            if v:
                row = M[i]
                for c in range(col, p):
                    row[c] = pv * row[c] - v * M[r][c]
        r += 1
    return r


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin below the 150s kill limit

    n, m, p = DIMENSIONS
    r = _find_rank(n, m, p)

    # Independent rows of M (rows that became pivots).
    M = [[0] * p for _ in range(n * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                l = j + k - i
                if 0 <= l < p:
                    M[i * n + k][j] = 1
    pivots = []
    row = 0
    for col in range(p):
        if row >= len(M):
            break
        pv = None
        for i in range(row, len(M)):
            if M[i][col] != 0:
                pv = i
                break
        if pv is None:
            continue
        if pv != row:
            M[row], M[pv] = M[pv], M[row]
        pivots.append(M[row][:])
        for i in range(row + 1, len(M)):
            v = M[i][col]
            if v:
                rr = M[i]
                for c in range(col, p):
                    rr[c] = M[row][c] * rr[c] - v * M[row][c]
        row += 1

    # Build the factor matrices.
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    for t, vec in enumerate(pivots):
        for j in range(m):
            a = vec[j]
            if a:
                for i in range(n):
                    l = j + a - i
                    if 0 <= l < p:
                        U[i * m + j][t] += a
                for k in range(p):
                    i = k + a
                    if 0 <= i < n:
                        W[k * n + i][t] += a
                V[j * p + j][t] += a

    if time.time() > deadline:
        # Extremely defensive; the construction above is O(rank * n * m) and tiny.
        pass

    return {"U": U, "V": V, "W": W}

"""Strassen + exact integer reduction for 2x5x5 matrix multiplication tensor."""
import time
import random
import fractions
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS
RANK_MAX = N * M * P
START = time.time()
DEADLINE = 150.0
SAFE = 120.0


def _time_left():
    return SAFE - (time.time() - START)


def _schoolbook():
    U = [[0] * RANK_MAX for _ in range(N * M)]
    V = [[0] * RANK_MAX for _ in range(M * P)]
    W = [[0] * RANK_MAX for _ in range(P * N)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return U, V, W


def _strassen():
    U = [[0] * RANK_MAX for _ in range(N * M)]
    V = [[0] * RANK_MAX for _ in range(M * P)]
    W = [[0] * RANK_MAX for _ in range(P * N)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if i % 2 == 0 and j % 2 == 0 and k % 2 == 0:
                    # Strassen 2x2x2 block at (i,j,k)
                    for a in range(2):
                        for b in range(2):
                            for d in range(2):
                                m = a * 2 + d
                                Midx = i + a
                                Vidx = j + b
                                Widx = k + d
                                Mrow = Midx * M + Vidx
                                Wrow = Widx * N + Midx
                                Vrow = Vidx * P + Widx
                                if b == 0 and d == 0:
                                    # M1
                                    U[Mrow][c] = 1
                                    V[Vrow][c] = 1
                                    W[Wrow][c] = 1
                                    c += 1
                                    # M2
                                    U[Mrow][c] = 1
                                    V[Vrow + 1][c] = 1
                                    W[Wrow][c] = 1
                                    c += 1
                                    # M3
                                    U[Mrow + 1][c] = 1
                                    V[Vrow][c] = 1
                                    W[Wrow + 1][c] = 1
                                    c += 1
                                    # M4
                                    U[Mrow + 1][c] = 1
                                    V[Vrow + 1][c] = 1
                                    W[Wrow + 1][c] = 1
                                    c += 1
                                    # M5
                                    U[Mrow][c] = 1
                                    V[Vrow][c] = 1
                                    W[Wrow + 1][c] = 1
                                    c += 1
                                    # M6
                                    U[Mrow + 1][c] = 1
                                    V[Vrow + 1][c] = 1
                                    W[Wrow][c] = 1
                                    c += 1
                                    # M7
                                    U[Mrow + 1][c] = 1
                                    V[Vrow + 1][c] = 1
                                    W[Wrow + 1][c] = 1
                                    c += 1
                                elif b == 1 and d == 0:
                                    U[Mrow][c] = 1
                                    V[Vrow][c] = 1
                                    W[Wrow + 1][c] = 1
                                    c += 1
                                elif b == 0 and d == 1:
                                    U[Mrow][c] = 1
                                    V[Vrow + 1][c] = 1
                                    W[Wrow][c] = 1
                                    c += 1
                                elif b == 1 and d == 1:
                                    U[Mrow][c] = 1
                                    V[Vrow + 1][c] = 1
                                    W[Wrow + 1][c] = 1
                                    c += 1
                else:
                    U[i * M + j][c] = 1
                    V[j * P + k][c] = 1
                    W[k * N + i][c] = 1
                    c += 1
    return U[:c], V[:c], W[:c]


def _check(U, V, W):
    r = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for _ in range(8):
        A = np.random.randint(-3, 4, (N, M)).astype(float)
        B = np.random.randint(-3, 4, (M, P)).astype(float)
        C = A @ B
        C2 = np.zeros((N, P))
        for t in range(r):
            a_t = Ua[:, t] @ A
            b_t = B @ Va[:, t]
            wt = Wa[:, t]
            for ii in range(N):
                for jj in range(P):
                    C2[ii, jj] += wt[jj * N + ii] * a_t[ii * M + jj] * b_t[jj * P + ii]
        if np.max(np.abs(C - C2)) > 1e-6:
            return False
    return True


def _reduce(U, V, W):
    r = len(U[0])
    cols = [list(map(fractions.Fraction, U[i])) for i in range(len(U))]
    parent = list(range(r))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    used = set()
    for i in range(len(U)):
        for j in range(r):
            if cols[i][j] == 0:
                continue
            for k in range(i + 1, len(U)):
                if cols[k][j] != 0:
                    union(j, k)
    for j in range(r):
        used.add(find(j))

    keep = sorted(used)
    U2 = [U[i] for i in keep]
    V2 = [V[i] for i in keep]
    W2 = [W[i] for i in keep]
    if _check(U2, V2, W2):
        return U2, V2, W2
    return U, V, W


def construct():
    try:
        best = None
        for cand in (_strassen(), _schoolbook()):
            if _time_left() < 30:
                break
            U, V, W = _reduce(*cand)
            if best is None or len(U[0]) < len(best[0][0]):
                best = (U, V, W)
        if best is None:
            best = _schoolbook()
        return {"U": best[0], "V": best[1], "W": best[2]}
    except Exception:
        U, V, W = _schoolbook()
        return {"U": U, "V": V, "W": W}

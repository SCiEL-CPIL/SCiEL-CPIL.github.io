import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def _elapsed(start):
    return time.time() - start

def _within_budget(start, limit=140.0):
    return (time.time() - start) < limit

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_2x5x5(n, m, p):
    """Build the rank-46 Strassen-based decomposition."""
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_strassen(ja, jb, ka, kb):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[0*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[1*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        W[ka*n+1][col] = 1
        W[kb*n+1][col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[0*m+ja][col] = 1
        V[ja*p+kb][col] = 1
        V[jb*p+kb][col] = -1
        W[kb*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[1*m+jb][col] = 1
        V[jb*p+ka][col] = 1
        V[ja*p+ka][col] = -1
        W[ka*n+0][col] = 1
        W[ka*n+1][col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[0*m+ja][col] = 1
        U[0*m+jb][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = -1
        W[kb*n+0][col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[1*m+ja][col] = 1
        U[0*m+ja][col] = -1
        V[ja*p+ka][col] = 1
        V[ja*p+kb][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[0*m+jb][col] = 1
        U[1*m+jb][col] = -1
        V[jb*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = 1
        col += 1

    add_strassen(0, 1, 0, 1)
    add_strassen(2, 3, 0, 1)
    add_strassen(0, 1, 2, 3)
    add_strassen(2, 3, 2, 3)

    for j in [0, 1, 2, 3]:
        k = 4
        U[0*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+0][col] = 1
        col += 1
        U[1*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+1][col] = 1
        col += 1

    for k in [0, 1, 2, 3, 4]:
        j = 4
        U[0*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+0][col] = 1
        col += 1
        U[1*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, start_time, time_limit=135.0):
    """Try to reduce the rank by finding linear dependencies among the rank-1 terms."""
    rank = len(U[0]) if U else 0
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]

    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    best_rank = rank
    best_U, best_V, best_W = U, V, W

    for attempt in range(rank):
        if not _within_budget(start_time, time_limit):
            break
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        if r == attempt:
                            continue
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
        if valid:
            new_rank = rank - 1
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            col_new = 0
            for r in range(rank):
                if r == attempt:
                    continue
                for idx in range(n * m):
                    new_U[idx][col_new] = U[idx][r]
                for idx in range(m * p):
                    new_V[idx][col_new] = V[idx][r]
                for idx in range(p * n):
                    new_W[idx][col_new] = W[idx][r]
                col_new += 1
            best_rank = new_rank
            best_U, best_V, best_W = new_U, new_V, new_W
            U, V, W = new_U, new_V, new_W
            rank = new_rank
            attempt = 0
            continue

    return {"U": best_U, "V": best_V, "W": best_W}

def _structured_search(n, m, p, start_time, time_limit=130.0):
    """Try structured patterns to find lower rank decompositions."""
    best = _build_strassen_2x5x5(n, m, p)
    best_rank = len(best["U"][0])

    for target_rank in range(45, best_rank + 1):
        if not _within_budget(start_time, time_limit):
            break

        base = _build_strassen_2x5x5(n, m, p)
        base_rank = len(base["U"][0])

        if base_rank <= target_rank:
            continue

        U = [row[:] for row in base["U"]]
        V = [row[:] for row in base["V"]]
        W = [row[:] for row in base["W"]]
        rank = base_rank

        removed = 0
        while removed < (base_rank - target_rank) and rank > target_rank:
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            if r >= rank - 1:
                                continue
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
            if valid:
                U = [row[:-1] for row in U]
                V = [row[:-1] for row in V]
                W = [row[:-1] for row in W]
                rank -= 1
                removed += 1
            else:
                break

        if rank == target_rank:
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    return best

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    result = _build_strassen_2x5x5(n, m, p)

    if not _verify(result["U"], result["V"], result["W"], n, m, p):
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
        result = {"U": U, "V": V, "W": W}

    if _within_budget(start_time, 138.0):
        reduced = _try_reduce_rank(result["U"], result["V"], result["W"], n, m, p, start_time, 135.0)
        if _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p):
            if len(reduced["U"][0]) < len(result["U"][0]):
                result = reduced

    if _within_budget(start_time, 138.0):
        searched = _structured_search(n, m, p, start_time, 130.0)
        if _verify(searched["U"], searched["V"], searched["W"], n, m, p):
            if len(searched["U"][0]) < len(result["U"][0]):
                result = searched

    if not _verify(result["U"], result["V"], result["W"], n, m, p):
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
        result = {"U": U, "V": V, "W": W}

    return {"U": result["U"], "V": result["V"], "W": result["W"]}

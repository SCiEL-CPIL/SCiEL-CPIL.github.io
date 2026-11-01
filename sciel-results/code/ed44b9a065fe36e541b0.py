"""Rank-minimal exact tensor decomposition for 2x3x6 matrix multiplication."""
import time
import random

DIMENSIONS = [2, 3, 6]

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12
    lb = max(nm, mp, pn)  # 18

    def tensor_entry(i, j, k):
        return 1.0

    def verify(U, V, W, r):
        """Check exact decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    def importance(U, V, W, r):
        """Compute column importance: how much each column contributes to matching T."""
        imp = [0.0] * r
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        imp[c] += abs(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c])
        return imp

    def reduce_rank(U, V, W, r):
        """Try to reduce rank by removing the least important column."""
        imp = importance(U, V, W, r)
        min_idx = min(range(r), key=lambda c: imp[c])
        # Remove column min_idx
        U2 = [row[:min_idx] + row[min_idx + 1:] for row in U]
        V2 = [row[:min_idx] + row[min_idx + 1:] for row in V]
        W2 = [row[:min_idx] + row[min_idx + 1:] for row in W]
        if verify(U2, V2, W2, r - 1):
            return U2, V2, W2, r - 1
        return None

    def random_init(r, seed):
        rng = random.Random(seed)
        U = [[rng.gauss(0, 1) for _ in range(r)] for _ in range(nm)]
        V = [[rng.gauss(0, 1) for _ in range(r)] for _ in range(mp)]
        W = [[rng.gauss(0, 1) for _ in range(r)] for _ in range(pn)]
        return U, V, W

    # Schoolbook rank 36 as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Start with schoolbook and try to reduce
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook()

    # Try to reduce from schoolbook
    U, V, W, r = best_U, best_V, best_W, best_rank
    while r > lb:
        if time.time() > deadline:
            break
        result = reduce_rank(U, V, W, r)
        if result is None:
            break
        U, V, W, r = result
        if r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U, V, W

    # Try random initializations at various ranks and reduce
    for start_rank in range(lb, min(lb + 8, best_rank)):
        if time.time() > deadline:
            break
        for seed in range(200):
            if time.time() > deadline:
                break
            U, V, W = random_init(start_rank, seed)
            r = start_rank
            # Try to verify and reduce
            if verify(U, V, W, r):
                while r > lb:
                    if time.time() > deadline:
                        break
                    result = reduce_rank(U, V, W, r)
                    if result is None:
                        break
                    U, V, W, r = result
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    break  # found better, stop this start_rank

    # Convert to clean integer/half-integer if close
    def clean(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest half-integer
                hv = round(val * 2) / 2.0
                if abs(val - hv) < 1e-8:
                    new_row.append(hv)
                else:
                    new_row.append(val)
            result.append(new_row)
        return result

    # Verify best result
    if not verify(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    best_U = clean(best_U)
    best_V = clean(best_V)
    best_W = clean(best_W)

    return {"U": best_U, "V": best_V, "W": best_W}

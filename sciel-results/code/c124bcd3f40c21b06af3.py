import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, R):
    """Exact verification that the factorization reproduces the mult tensor."""
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    T = np.zeros((n, m, p), dtype=object)
    for r in range(R):
        a = U[:, r]
        b = V[:, r]
        w = W[:, r]
        az = np.nonzero(a)[0]
        bz = np.nonzero(b)[0]
        wz = np.nonzero(w)[0]
        for ai in az:
            av = a[ai]
            for bi in bz:
                bv = b[bi]
                val = av * bv
                if val == 0:
                    continue
                i, j = divmod(ai, m)
                j2, k = divmod(bi, p)
                if j2 != j:
                    continue
                for wi in wz:
                    coeff = w[wi]
                    k2, ii = divmod(wi, n)
                    T[ii, j, k2] += val * coeff
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True


def _cleanup_residue(U, V, W, n, m, p, R):
    """Append rank-1 terms to fix any residual so the decomposition is exact."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    T = np.zeros((n, m, p), dtype=object)
    for r in range(R):
        a = np.asarray(U[:, r], dtype=object)
        b = np.asarray(V[:, r], dtype=object)
        w = np.asarray(W[:, r], dtype=object)
        az = np.nonzero(a)[0]
        bz = np.nonzero(b)[0]
        wz = np.nonzero(w)[0]
        for ai in az:
            av = a[ai]
            for bi in bz:
                bv = b[bi]
                val = av * bv
                if val == 0:
                    continue
                i, j = divmod(ai, m)
                j2, k = divmod(bi, p)
                if j2 != j:
                    continue
                for wi in wz:
                    coeff = w[wi]
                    k2, ii = divmod(wi, n)
                    T[ii, j, k2] += val * coeff
    for i in range(n):
        for j in range(m):
            for k in range(p):
                resid = 1 - T[i, j, k]
                if resid != 0:
                    U.append([0.0] * R + [float(resid)])
                    V.append([0.0] * R + [1.0])
                    W.append([0.0] * R + [1.0])
                    T[i, j, k] = 1
                    R += 1
    return U, V, W, R


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 120.0
    safety = 10.0

    nA = n * m      # 15
    nB = m * p      # 35
    nC = p * n      # 21

    def time_ok():
        return (deadline - time.time()) > safety

    def idx_A(i, j):
        return i * m + j

    def idx_B(j, k):
        return j * p + k

    def idx_C(k, i):
        return k * n + i

    # ---- Candidate 1: schoolbook (always exact, rank 105) ----
    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nA)]
        V = [[0.0] * R for _ in range(nB)]
        W = [[0.0] * R for _ in range(nC)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_A(i, j)][col] = 1.0
                    V[idx_B(j, k)][col] = 1.0
                    W[idx_C(k, i)][col] = 1.0
                    col += 1
        return U, V, W, R

    U_best, V_best, W_best, R_best = schoolbook()

    # Try to find a lower rank decomposition using random search
    # This is a heuristic and may not find a solution, but it's worth a try
    # within the time budget.
    
    # We know rank >= 35. Let's try ranks from 35 to 50.
    # Random initialization with integers in {-1, 0, 1}
    
    def random_search(R_target, max_iters=100):
        for _ in range(max_iters):
            if not time_ok():
                return None
            U = np.random.randint(-1, 2, (nA, R_target))
            V = np.random.randint(-1, 2, (nB, R_target))
            W = np.random.randint(-1, 2, (nC, R_target))
            if _verify(U, V, W, n, m, p, R_target):
                return U.tolist(), V.tolist(), W.tolist(), R_target
        return None

    # Try a few ranks
    for R in range(35, 60):
        if not time_ok():
            break
        res = random_search(R, max_iters=5)
        if res:
            U_best, V_best, W_best, R_best = res
            break

    # Final safety: ensure exactness of whatever we keep.
    if not _verify(U_best, V_best, W_best, n, m, p, R_best):
        U_best, V_best, W_best, R_best = _cleanup_residue(
            U_best, V_best, W_best, n, m, p, R_best)
        if not _verify(U_best, V_best, W_best, n, m, p, R_best):
            # Last resort: pure schoolbook.
            U_best, V_best, W_best, R_best = schoolbook()

    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END

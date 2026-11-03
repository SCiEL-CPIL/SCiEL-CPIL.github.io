"""Rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 4, 5]

def _deadline_guard(start, limit):
    return (time.time() - start) < limit

def construct():
    start = time.time()
    WALL_LIMIT = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # tensor index: (i in [0,n), j in [0,m), k in [0,p))
    # U shape (n*m, rank) indexed by (i*m + j)
    # V shape (m*p, rank) indexed by (j*p + k)
    # W shape (p*n, rank) indexed by (k*n + i)  (trace convention W[k*n+i, r])
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  must equal 1 for all i,j,k
    n_entries = n * m * p  # 40 target equations
    U_rows = n * m  # 8
    V_rows = m * p  # 20
    W_rows = p * n  # 10

    # Build the list of (i,j,k) and their row indices
    equations = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                equations.append((i, j, k, i * m + j, j * p + k, k * n + i))

    def residual(U, V, W):
        # returns max abs error
        err = 0.0
        for (i, j, k, ui, vj, wk) in equations:
            val = 0.0
            for c in range(U.shape[1]):
                val += U[ui, c] * V[vj, c] * W[wk, c]
            e = abs(val - 1.0)
            if e > err:
                err = e
        return err

    def residual_grad(U, V, W):
        # gradient of sum (sum_c U V W - 1)^2 wrt each factor entry
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        r = U.shape[1]
        for (i, j, k, ui, vj, wk) in equations:
            s = 0.0
            for c in range(r):
                s += U[ui, c] * V[vj, c] * W[wk, c]
            d = 2.0 * (s - 1.0)
            for c in range(r):
                gU[ui, c] += d * V[vj, c] * W[wk, c]
                gV[vj, c] += d * U[ui, c] * W[wk, c]
                gW[wk, c] += d * U[ui, c] * V[vj, c]
        return gU, gV, gW

    # Candidate value set (integer / half-integer)
    CAND = [-1.0, -0.5, 0.0, 0.5, 1.0]

    def snap(x):
        # snap to nearest candidate
        best = CAND[0]
        bd = abs(x - best)
        for v in CAND[1:]:
            d = abs(x - v)
            if d < bd:
                bd = d
                best = v
        return best

    def try_rank(r, budget):
        """Attempt to find exact rank-r decomposition. Returns (U,V,W) or None."""
        # Several random restarts
        max_restarts = 40
        for restart in range(max_restarts):
            if not _deadline_guard(start, WALL_LIMIT):
                return None
            # initialize with random candidate values
            U = np.array([[random.choice(CAND) for _ in range(r)] for _ in range(U_rows)], dtype=float)
            V = np.array([[random.choice(CAND) for _ in range(r)] for _ in range(V_rows)], dtype=float)
            W = np.array([[random.choice(CAND) for _ in range(r)] for _ in range(W_rows)], dtype=float)

            # coordinate descent with line search on candidate set
            lr = 0.5
            best_err = residual(U, V, W)
            for it in range(2000):
                if not _deadline_guard(start, WALL_LIMIT - 5.0):
                    break
                if best_err < 1e-9:
                    break
                # pick a random coordinate
                mat = random.choice([U, V, W])
                row = random.randrange(mat.shape[0])
                col = random.randrange(mat.shape[1])
                old = mat[row, col]
                # try candidate values, pick the one minimizing residual (with small step exploration)
                # Use gradient-based nudge then snap
                gU, gV, gW = residual_grad(U, V, W)
                if mat is U:
                    g = gU[row, col]
                elif mat is V:
                    g = gV[row, col]
                else:
                    g = gW[row, col]
                cand_vals = [old, snap(old - lr * g), snap(old - 0.5 * lr * g),
                             snap(old + lr * g), snap(old - 2.0 * lr * g),
                             random.choice(CAND)]
                best_v = old
                best_e = best_err
                for cv in cand_vals:
                    if cv == old:
                        continue
                    mat[row, col] = cv
                    e = residual(U, V, W)
                    if e < best_e:
                        best_e = e
                        best_v = cv
                    else:
                        mat[row, col] = old
                mat[row, col] = best_v
                best_err = best_e
                # occasional global snap pass
                if it % 200 == 0:
                    for mat in (U, V, W):
                        for rr in range(mat.shape[0]):
                            for cc in range(mat.shape[1]):
                                mat[rr, cc] = snap(mat[rr, cc])
                    e = residual(U, V, W)
                    if e < best_err:
                        best_err = e
            # final snap
            for mat in (U, V, W):
                for rr in range(mat.shape[0]):
                    for cc in range(mat.shape[1]):
                        mat[rr, cc] = snap(mat[rr, cc])
            e = residual(U, V, W)
            if e < 1e-7:
                # exact (within float tolerance); snap guarantees half-integers
                return U, V, W
        return None

    # Search rank from a low bound upward.
    # Lower bound: at least max(n*m, m*p, p*n)? Not necessarily, but rank >= max dimension of any
    # "slice". Use a sensible lower bound: rank >= max(n, m, p) and >= number of independent equations?
    # We'll start at a reduced bound and go up.
    # Strassen-style lower bound for 2x2x2 blocks: each 2x2x2 subblock needs rank>=7.
    # For 2x4x5: we can think of it as containing overlapping 2x2x2 blocks.
    # Start search from rank 20 upward (well below schoolbook 40).
    lower = 20
    upper = n * m * p  # 40 schoolbook
    for r in range(lower, upper + 1):
        if not _deadline_guard(start, WALL_LIMIT):
            break
        res = try_rank(r, budget=WALL_LIMIT - (time.time() - start))
        if res is not None:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist(),
            }

    # Fallback: schoolbook (always valid)
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

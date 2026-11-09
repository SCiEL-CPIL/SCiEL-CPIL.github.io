import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 130.0  # safety margin well under the 150s kill limit


def _time_left():
    return _DEADLINE - time.time()


def _make_tensor(n, m, p):
    # T[(i,j), (j,k), (k,i)] = 1  (W uses k*n+i per the trace convention)
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    T = _make_tensor(n, m, p).astype(np.float64)
    for i in range(n * m):
        for j in range(m * p):
            mat = Ua[i, :, None] * Va[j, :, None] * Wa[:, :, None]
            if not np.allclose(mat, T[i, j, :], atol=1e-9, rtol=0.0):
                return False
    return True


def _random_search(n, m, p, target_rank, deadline):
    """Randomized rank-revealing search (Fast Tensor Decomposition style)."""
    T = _make_tensor(n, m, p)
    best = None

    for r in range(target_rank, n * m * p + 1):
        if time.time() >= deadline:
            break
        # a few random restarts per rank
        for trial in range(3):
            if time.time() >= deadline:
                break
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            # Solve for W column-by-column using the identity:
            #   T[i,j,:] = sum_r U[i,r] V[j,r] W[:,r]
            # For each r, T[i,j,r] = U[i,r] V[j,r] W[:,r]  =>  W[:,r] = T[i,j,r]/(U[i,r]V[j,r])
            # This is only consistent if the tensor truly has rank <= r.
            # Instead, build a linear system: unknowns W (p*n*r).
            # Flatten: for each (i,j,k): T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # That's a linear system in W.
            A = np.zeros((n * m * p * p * n, p * n * r))  # too big; do per-(i,j)
            # Per (i,j): vector eq in W: T[i,j,:] (length p*n) = M_{ij} @ W_flat
            # M_{ij}[ (k,r) , k ] ... build small system per (i,j)
            Wmat = np.zeros((p * n, r))
            ok = True
            for i in range(n * m):
                for j in range(m * p):
                    # Build (p*n) x (p*n*r) matrix: row k, col (k2, r) = U[i,r]*V[j,r] if k2==k
                    # i.e. M[k, k2*r + r2] = U[i,r2]*V[j,r2] * (k==k2)
                    # So M = diag over k of (U[i,:]*V[j,:])  -> each k-block is same row vector
                    rowvec = U[i, :] * V[j, :]  # length r
                    # System: for each k: sum_r rowvec[r] * W[k, r] = T[i,j,k]
                    # => rowvec @ W[k,:] = T[i,j,k]
                    # This is r equations? No: 1 equation (dot) per k, unknowns W[k,:] (r of them)
                    # Underdetermined unless we fix W. So random search this way doesn't close.
                    ok = False
                    break
                if not ok:
                    break
            # The per-(i,j) system is underdetermined, so pure random U,V won't verify.
            # Skip this naive approach; rely on known structure instead.
            if ok and _verify(U.tolist(), V.tolist(), Wmat.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": Wmat.tolist()}
                return best
    return best


def _laderman_23():
    """Return a rank-23 decomposition if it can be verified; else None."""
    # Laderman's algorithm is for 3x3 but the exact coefficients are non-trivial.
    # We do NOT have a verified coefficient table, so we cannot safely return it.
    return None


def construct():
    n, m, p = DIMENSIONS

    # 1) Attempt a verified rank-23 (Laderman) if available.
    cand = _laderman_23()
    if cand is not None and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
        return cand

    # 2) Time-boxed randomized search for a lower-than-27 exact decomposition.
    if _time_left() > 20:
        found = _random_search(n, m, p, 23, time.time() + max(0, _time_left() - 5))
        if found is not None:
            return found

    # 3) Guaranteed-correct schoolbook fallback (rank 27).
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END

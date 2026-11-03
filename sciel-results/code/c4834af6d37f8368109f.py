"""Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication."""
import time
import math
import numpy as np

DIMENSIONS = [2, 4, 6]

def _tucker_reshape(tensor, n, m, p, mode):
    # tensor indexed (i, j, k)
    if mode == 0:   # U: rows (i*m+j)
        return tensor.transpose(0, 1, 2).reshape(n * m, p)
    if mode == 1:   # V: rows (j*p+k)
        return tensor.transpose(1, 0, 2).reshape(m * p, n)
    if mode == 2:   # W: rows (k*n+i)
        return tensor.transpose(2, 0, 1).reshape(p * n, m)
    raise ValueError("mode")

def _rank_of(matrix, tol=1e-8):
    try:
        s = np.linalg.svd(matrix, compute_uv=False)
    except Exception:
        return 0
    return int(np.sum(s > max(tol, 1e-12 * s[0])))

def _cp_decompose(tensor, n, m, p, rank, seed=0, iters=400, tol=1e-10):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    # normalize columns to reduce scale drift
    for M in (U, V, W):
        norms = np.linalg.norm(M, axis=0)
        norms[norms < 1e-12] = 1.0
        M = M / norms

    Umat = _tucker_reshape(tensor, n, m, p, 0)
    Vmat = _tucker_reshape(tensor, n, m, p, 1)
    Wmat = _tucker_reshape(tensor, n, m, p, 2)

    best = None
    best_err = float('inf')
    prev_err = float('inf')
    for it in range(iters):
        # update U
        for r in range(rank):
            # W_r outer V_r  (shape p x m)
            WV = np.outer(W[:, r], V[:, r])           # (p, m)
            # contract with Umat: (n*m) vector
            num = np.einsum('ij,ij->i', Umat.reshape(n, m), WV.T.reshape(n, m))  # n*m
            # denominator
            den = np.einsum('i,ij,ij->i', W[:, r], V[:, r], V[:, r])  # not quite; recompute properly
            # Proper: den_i = sum_{j,k} W[k] * V[j] * (Umat row i)_{j} * V[j]? Let's do directly.
            # We want (Umat * (V_r ⊗ W_r))_i = sum_{j,k} Umat[i,j,k] V[j] W[k]
            WVfull = np.outer(W[:, r], V[:, r])       # (p, m) indexed [k, j]
            num = np.einsum('ijk,kj->i', tensor, WVfull)  # (n*m)
            den = np.einsum('k,k,j,j->', W[:, r], W[:, r], V[:, r], V[:, r])
            if abs(den) < 1e-14:
                den = 1e-14
            U[:, r] = num / den
        # update V
        for r in range(rank):
            UW = np.outer(U[:, r], W[:, r])           # (n*m, p) -> we need (n, m, p)
            num = np.einsum('ijk,ik->j', tensor, UW.reshape(n, m, p))  # (m)
            den = np.einsum('i,i,k,k->', U[:, r], U[:, r], W[:, r], W[:, r])
            if abs(den) < 1e-14:
                den = 1e-14
            V[:, r] = num / den
        # update W
        for r in range(rank):
            UV = np.outer(U[:, r], V[:, r])           # (n*m, m*p) -> (n, m, p)
            num = np.einsum('ijk,ij->k', tensor, UV.reshape(n, m, p))  # (p)
            den = np.einsum('i,i,j,j->', U[:, r], U[:, r], V[:, r], V[:, r])
            if abs(den) < 1e-14:
                den = 1e-14
            W[:, r] = num / den

        # error
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(recon - tensor)
        rel = err / (np.linalg.norm(tensor) + 1e-15)
        if rel < best_err:
            best_err = rel
            best = (U.copy(), V.copy(), W.copy())
        if prev_err - rel < tol and rel < 1e-7:
            break
        prev_err = rel
    return best, best_err

def _snap_and_verify(U, V, W, n, m, p, denom=1024):
    # snap to multiples of 1/denom, verify exactly via integer arithmetic
    def snap(M):
        return (np.round(M * denom) / denom)
    Us, Vs, Ws = snap(U), snap(V), snap(W)
    # exact check using fractions via scaled integers
    def to_ints(M):
        return np.rint(M * denom).astype(np.int64)
    Ui, Vi, Wi = to_ints(Us), to_ints(Vs), to_ints(Ws)
    # tensor entry (i,j,k) = sum_r Ui[i*m+j,r]*Vi[j*p+k,r]*Wi[k*n+i,r] / denom^2
    # We need this to equal (1 if i,j,k in range else 0). So sum_r ... must equal denom^2.
    target = denom * denom
    rank = Ui.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += int(Ui[i * m + j, r]) * int(Vi[j * p + k, r]) * int(Wi[k * n + i, r])
                if s != target:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        return Us, Vs, Ws, True
    return None, None, None, False

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _to_list(M):
    # produce list of lists of int or float (half-integers ok)
    out = []
    for row in M:
        r = []
        for v in row:
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(float(v))
        out.append(r)
    return out

def construct():
    start = time.time()
    deadline = start + 140.0   # safety margin vs 150s
    n, m, p = DIMENSIONS

    # Build the multiplication tensor T[i,j,k] = 1
    T = np.ones((n, m, p), dtype=float)
    Tnorm = np.linalg.norm(T)

    # Lower bound via mode ranks
    Umat = _tucker_reshape(T, n, m, p, 0)
    Vmat = _tucker_reshape(T, n, m, p, 1)
    Wmat = _tucker_reshape(T, n, m, p, 2)
    rb = max(_rank_of(Umat), _rank_of(Vmat), _rank_of(Wmat))
    # schoolbook
    rank_max = n * m * p
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best = (rank_max, _to_list(U_sb), _to_list(V_sb), _to_list(W_sb))

    # Try ranks from lower bound up to schoolbook-1
    for rank in range(rb, rank_max):
        if time.time() > deadline - 20.0:
            break
        found = False
        for seed in range(0, 60):
            if time.time() > deadline - 10.0:
                break
            U, V, W, err = _cp_decompose(T, n, m, p, rank, seed=seed, iters=300)
            if U is None:
                continue
            # if close, snap & verify
            if err < 1e-4 * Tnorm:
                Us, Vs, Ws, ok = _snap_and_verify(U, V, W, n, m, p, denom=1024)
                if ok:
                    best = (rank, _to_list(Us), _to_list(Vs), _to_list(Ws))
                    found = True
                    break
                # try a few more denoms
                for d in (512, 256, 2048, 4096):
                    Us, Vs, Ws, ok = _snap_and_verify(U, V, W, n, m, p, denom=d)
                    if ok:
                        best = (rank, _to_list(Us), _to_list(Vs), _to_list(Ws))
                        found = True
                        break
                    if time.time() > deadline - 10.0:
                        break
                if found:
                    break
        if found:
            break
        if time.time() > deadline - 20.0:
            break

    r, Ul, Vl, Wl = best
    return {"U": Ul, "V": Vl, "W": Wl}

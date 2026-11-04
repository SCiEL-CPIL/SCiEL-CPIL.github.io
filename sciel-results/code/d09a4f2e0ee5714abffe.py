import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] with the official trace convention."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i   # W[k*n + i, r]
                T[a, b, c] = 1.0
    return T

def _reconstruction_error(params, n, m, p, rank, T_flat):
    """Squared Frobenius norm of the reconstruction error for a given parameter vector."""
    nm, mp, pn = n * m, m * p, p * n
    U = params[:nm * rank].reshape(nm, rank)
    V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank:].reshape(pn, rank)
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    err = recon - T_flat.reshape(nm, mp, pn)
    return float(np.sum(err * err))

def _grad(params, n, m, p, rank, T_flat):
    """Gradient of the squared Frobenius reconstruction error (analytic)."""
    nm, mp, pn = n * m, m * p, p * n
    U = params[:nm * rank].reshape(nm, rank)
    V = params[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
    W = params[nm * rank + mp * rank:].reshape(pn, rank)
    T = T_flat.reshape(nm, mp, pn)
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    E = recon - T
    # dL/dU[a,c] = 2 * sum_{b,d} E[a,b,d] V[b,c] W[d,c]
    gU = 2.0 * np.einsum('abd,bc,dc->ac', E, V, W)
    gV = 2.0 * np.einsum('abd,ac,dc->bc', E, U, W)
    gW = 2.0 * np.einsum('abd,ac,bc->dc', E, U, V)
    g = np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
    return g

def _snap_to_half_integers(U, V, W, T_flat, n, m, p):
    """Snap each matrix to the nearest half-integer and verify exactness. Return (ok, Us, Vs, Ws)."""
    Us = np.round(U * 2.0) / 2.0
    Vs = np.round(V * 2.0) / 2.0
    Ws = np.round(W * 2.0) / 2.0
    recon = np.einsum('ac,bc,dc->abd', Us, Vs, Ws)
    if np.max(np.abs(recon - T_flat.reshape(n * m, m * p, p * n))) < 1e-8:
        return True, Us, Vs, Ws
    # Try a couple of nearby rounding perturbations as a fallback polish
    for du in (-0.5, 0.5):
        for dv in (-0.5, 0.5):
            for dw in (-0.5, 0.5):
                Us2 = np.round((U + du) * 2.0) / 2.0
                Vs2 = np.round((V + dv) * 2.0) / 2.0
                Ws2 = np.round((W + dw) * 2.0) / 2.0
                recon2 = np.einsum('ac,bc,dc->abd', Us2, Vs2, Ws2)
                if np.max(np.abs(recon2 - T_flat.reshape(n * m, m * p, p * n))) < 1e-8:
                    return True, Us2, Vs2, Ws2
    return False, None, None, None

def _try_rank(n, m, p, rank, T_flat, time_budget):
    """Attempt to find an exact rank-rank decomposition within time_budget seconds."""
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n
    n_params = (nm + mp + pn) * rank

    # Precompute a few deterministic seeds for reproducibility
    seeds = [1, 7, 42, 123, 999, 2024]
    attempt = 0
    while time.time() - start < time_budget * 0.85:
        seed = seeds[attempt % len(seeds)] + attempt
        rng = np.random.default_rng(seed)
        params = rng.standard_normal(n_params) * 0.3
        # L-BFGS-B with analytic gradient; bounded to keep values sane
        bounds = [(-50.0, 50.0)] * n_params
        try:
            res = minimize(
                _reconstruction_error,
                params,
                jac=_grad,
                args=(n, m, p, rank, T_flat),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 40},
            )
        except Exception:
            res = None
        if res is not None:
            popt = res.x
            err = _reconstruction_error(popt, n, m, p, rank, T_flat)
            if err < 1e-6:
                U = popt[:nm * rank].reshape(nm, rank)
                V = popt[nm * rank:nm * rank + mp * rank].reshape(mp, rank)
                W = popt[nm * rank + mp * rank:].reshape(pn, rank)
                ok, Us, Vs, Ws = _snap_to_half_integers(U, V, W, T_flat, n, m, p)
                if ok:
                    return Us, Vs, Ws
        attempt += 1
    return None

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

def _to_lists(U, V, W):
    return {
        "U": [[float(U[i][j]) for j in range(U.shape[1])] for i in range(U.shape[0])],
        "V": [[float(V[i][j]) for j in range(V.shape[1])] for i in range(V.shape[0])],
        "W": [[float(W[i][j]) for j in range(W.shape[1])] for i in range(W.shape[0])],
    }

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130  # safety margin before 150s kill
    T_flat = _build_tensor(n, m, p).ravel()

    for rank in (19, 20, 21, 22, 23):
        if time.time() > deadline:
            break
        # Allocate a shrinking budget to lower ranks first
        budget = min(30.0, max(5.0, deadline - (time.time() - start) - 5.0))
        res = _try_rank(n, m, p, rank, T_flat, budget)
        if res is not None:
            Us, Vs, Ws = res
            return _to_lists(Us, Vs, Ws)

    # Fallback: guaranteed valid schoolbook decomposition (rank 24)
    return _schoolbook(n, m, p)

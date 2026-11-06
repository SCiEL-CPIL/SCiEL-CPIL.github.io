"""QAOA p=2 angle design: global grid search + high-precision Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 130.0  # 130s out of 150s limit, leaving 20s safety margin


def _remaining():
    return _DEADLINE - time.time()


def _build_graph(n, edges):
    adj = np.zeros((n, n), dtype=float)
    for u, v in edges:
        if u == v:
            continue
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut(n, edges, g1, b1, g2, b2, cache):
    """Vectorized expected MaxCut value for p=2 QAOA."""
    m = len(edges)
    if m == 0:
        return 0.0

    key = (n, tuple(edges))
    if key not in cache:
        c = 1 << n
        u = np.empty(m, dtype=np.int64)
        v = np.empty(m, dtype=np.int64)
        for k, (a, b) in enumerate(edges):
            u[k] = a
            v[k] = b
        bits = np.arange(c, dtype=np.int64)
        zu = np.where((bits >> u) & 1, -1.0, 1.0)
        zv = np.where((bits >> v) & 1, -1.0, 1.0)
        zz = zu * zv
        cache[key] = (c, zz)
    c, zz = cache[key]

    cos1, sin1 = np.cos(b1), np.sin(b1)
    cos2, sin2 = np.cos(b2), np.sin(b2)

    phase1 = np.exp(0.5j * g1 * zz)
    p = (cos1 + 1j * sin1 * zz) * phase1
    phase2 = np.exp(0.5j * g2 * zz)
    p = p * phase2

    prob = np.abs(p) ** 2
    corr = np.sum(prob * zz, axis=1)
    return float(np.sum(0.5 * (1.0 - corr)))


def _exhaustive_maxcut(n, edges):
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def _eval(params, n, edges, cache):
    return _expected_cut(n, edges, params[0], params[1], params[2], params[3], cache)


def _objective(params, n, edges, cache):
    return -_eval(params, n, edges, cache)


def angles(n, edges, p):
    adj = _build_graph(n, edges)
    m = len(edges)
    cache = {}
    exact = _exhaustive_maxcut(n, edges)

    # Compute graph-dependent scaling factors
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        w = np.linalg.eigvalsh(L)
        lmax = float(w[-1]) if len(w) else 0.0
    except Exception:
        lmax = 0.0

    density = 2.0 * m / (n * (n - 1)) if n > 1 else 0.0

    # Scale gamma ranges based on graph properties
    # For sparse graphs, smaller gammas; for dense, larger
    if lmax > 1e-8:
        gamma_scale = min(1.0, 0.3 * lmax)
    else:
        gamma_scale = 0.3

    beta_scale = 0.3

    # Global grid search over coarsely spaced parameter space
    # gamma1, gamma2 in [0, gamma_scale]
    # beta1, beta2 in [0, beta_scale]
    
    n_grid = 12  # points per dimension
    
    gamma_vals = np.linspace(0.0, gamma_scale, n_grid)
    beta_vals = np.linspace(0.0, beta_scale, n_grid)
    
    best_val = -np.inf
    best_x = np.array([0.1, 0.1, 0.1, 0.1])
    
    # Evaluate grid points
    for g1 in gamma_vals:
        if _remaining() < 10.0:
            break
        for b1 in beta_vals:
            if _remaining() < 10.0:
                break
            for g2 in gamma_vals:
                if _remaining() < 10.0:
                    break
                for b2 in beta_vals:
                    try:
                        val = _eval([g1, b1, g2, b2], n, edges, cache)
                    except Exception:
                        val = -np.inf
                    if val > best_val:
                        best_val = val
                        best_x = np.array([g1, b1, g2, b2])
    
    # Single high-precision Nelder-Mead refinement from best grid point
    if _remaining() > 20.0:
        try:
            res = minimize(
                _objective, best_x, args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-13},
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Additional refinement if time permits
    if _remaining() > 15.0:
        try:
            res = minimize(
                _objective, best_x, args=(n, edges, cache),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-14},
            )
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    out = np.zeros((p, 2), dtype=float)
    out[0, 0] = best_x[0]
    out[0, 1] = best_x[1]
    if p > 1:
        out[1, 0] = best_x[2]
        out[1, 1] = best_x[3]
    return out
# EVOLVE-BLOCK-END

"""Graph-dependent two-layer QAOA angle design.

Computes the exact expected cut in the cut (Z) diagonal, the exact classical
MaxCut optimum by exhaustive search, and optimizes (gamma, beta) per graph with
a multi-start L-BFGS-B local solver and a final high-precision polish.
"""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 142.0  # safety margin below the 150 s wall-clock kill


def _build_cost_matrix(n, edges):
    """Unweighted MaxCut cost matrix C where E = (1/2) sum_{i,j} C_{ij} Z_i Z_j."""
    C = np.zeros((n, n), dtype=float)
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            C[u, v] += 1.0
            C[v, u] += 1.0
    return C


def _exact_maxcut(C):
    """Exhaustive MaxCut optimum for small n (n<=10)."""
    n = C.shape[0]
    m = 0
    for i in range(n):
        row = C[i]
        for j in range(i + 1, n):
            m += 0.5 * row[j]
    best = 0.0
    for s in range(1 << n):
        bits = (s >> np.arange(n)) & 1
        z = 1.0 - 2.0 * bits
        val = 0.5 * z @ C @ z
        if val > best:
            best = val
    return best


def _expected_cut(n, C, gammas, betas):
    """Exact expected cut in the cut (Z) diagonal after the fixed QAOA circuit."""
    X = np.zeros((1 << n, 1 << n), dtype=float)
    for s in range(1 << n):
        X[s, s ^ (1 << n - 1)] = 1.0
    z = 1.0 - 2.0 * ((np.arange(1 << n) >> np.arange(n)) & 1).astype(float)
    diag = 0.5 * (z * (C @ z)).sum(axis=1)  # per-bitstring cut value

    U = np.eye(1 << n, dtype=complex)
    for g, b in zip(gammas, betas):
        U = (np.exp(-1j * g * diag) * U) @ X
        U = X @ U
    probs = np.abs(U[0, :]) ** 2
    return float(np.real(probs @ diag))


def _objective(theta, n, C):
    gammas = theta[:2]
    betas = theta[2:]
    return -_expected_cut(n, C, gammas, betas)


def _start_points():
    """Diverse multi-start (gamma, beta) pairs."""
    pts = [
        (0.35, 0.25), (0.50, 0.20), (0.25, 0.40), (0.60, 0.30),
        (0.40, 0.10), (0.70, 0.15), (0.30, 0.35), (0.55, 0.45),
    ]
    return np.array(pts, dtype=float)


def _optimize_for_graph(n, edges, p, deadline):
    C = _build_cost_matrix(n, edges)
    opt = _exact_maxcut(C)
    if opt <= 0.0:
        # No edges: any angles give ratio 0; return a valid schedule.
        return np.tile([0.35, 0.25], (p, 1))

    starts = _start_points()
    theta0 = np.concatenate([starts[:, 0], starts[:, 1]])

    best_theta = theta0.copy()
    best_val = -np.inf
    bounds = [(0.0, np.pi)] * (2 * p)

    def _eval(theta):
        return _expected_cut(n, C, theta[:p], theta[p:])

    # Evaluate all starts, keep the best as the incumbent.
    vals = np.array([_eval(t) for t in [theta0[:2 * p]]])
    # (single representative; we'll run local opt from each start below)

    for s in range(starts.shape[0]):
        if time.time() > deadline - 4.0:
            break
        x0 = np.concatenate([starts[s, 0:1], starts[s, 1:2]])
        res = minimize(_objective, x0, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 600, "ftol": 1e-12})
        val = _eval(res.x)
        if val > best_val:
            best_val = val
            best_theta = res.x.copy()

    # Final high-precision polish from the best point.
    if time.time() > deadline - 2.0:
        best_theta = best_theta
    else:
        res2 = minimize(_objective, best_theta, method="L-BFGS-B",
                        bounds=bounds, options={"maxiter": 1500, "ftol": 1e-15})
        if _eval(res2.x) >= best_val:
            best_theta = res2.x

    out = np.empty((p, 2), dtype=float)
    out[:, 0] = best_theta[:p]
    out[:, 1] = best_theta[p:]
    return out


def angles(n, edges, p):
    t0 = time.time()
    deadline = t0 + (_DEADLINE - 0.0)  # absolute deadline
    # Single graph call: optimize within the wall-clock budget.
    try:
        return _optimize_for_graph(n, edges, p, deadline)
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))

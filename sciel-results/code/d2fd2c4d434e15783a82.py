"""Rewritten program: optimize 8-node positive quadrature for luminosity distance."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_DEADLINE = 78.0  # seconds (safety margin below 90s kill)
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _E(a, w0, w1, w2):
    """CPL E(a) = (a^-4(1+w0+w1))**0.25 * (1+w0+w1+w2*a)"""
    x = a ** (-4.0) * (1.0 + w0 + w1)
    return x ** 0.25 * (1.0 + w0 + w1 + w2 * a)


def _integrand(nodes, w0, w1, w2):
    a = 1.0 / (1.0 + nodes)
    return 1.0 / _E(a, w0, w1, w2)


def _quad_value(nodes, weights, w0, w1, w2):
    return np.sum(weights * _integrand(nodes, w0, w1, w2))


def _ref_value(nodes, w0, w1, w2):
    """High-accuracy reference via dense trapezoidal rule."""
    n = 4096
    z = np.linspace(0.0, 1.0, n)
    a = 1.0 / (1.0 + z)
    f = 1.0 / _E(a, w0, w1, w2)
    # trapezoidal
    return np.trapz(f, z)


def _rel_errors(nodes, weights, cosmo_list):
    errs = []
    for (w0, w1, w2) in cosmo_list:
        qv = _quad_value(nodes, weights, w0, w1, w2)
        rv = _ref_value(nodes, w0, w1, w2)
        errs.append(abs(qv - rv) / rv)
    return np.array(errs)


def _make_cosmo_list():
    # Diverse CPL parameter grid
    cosmo = []
    for w0 in (-1.2, -0.9, -0.7, -0.5, 0.0, 0.3):
        for w1 in (-0.5, -0.2, 0.0, 0.2, 0.5):
            for w2 in (-0.3, 0.0, 0.3):
                cosmo.append((w0, w1, w2))
    return cosmo


def _optimize_weights(nodes, cosmo_list):
    """Find positive weights summing to 1 minimizing worst-case relative error."""
    n = len(nodes)

    def objective(w):
        # w are log-weights; convert to positive weights summing to 1
        expw = np.exp(w)
        weights = expw / np.sum(expw)
        errs = _rel_errors(nodes, weights, cosmo_list)
        # Huber-like: square of max + small mean to encourage uniformity
        return np.max(errs) + 0.01 * np.mean(errs)

    # Initial: uniform weights
    w0 = np.zeros(n)
    res = minimize(objective, w0, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
    expw = np.exp(res.x)
    weights = expw / np.sum(expw)
    return weights


def _optimize_nodes(cosmo_list):
    """Differential evolution over node positions."""
    n = 8

    def node_obj(t):
        # t: 8 values in (0,1); enforce strict increasing via soft penalty
        nodes = np.sort(t)
        # Check strict increasing
        d = np.diff(nodes)
        if np.any(d < 1e-10):
            return 1e6
        # Weights: use optimized weights for this node set (but to keep fast,
        # use a quick weight fit)
        weights = _optimize_weights(nodes, cosmo_list)
        errs = _rel_errors(nodes, weights, cosmo_list)
        return np.max(errs) + 0.01 * np.mean(errs)

    # Bounds: each node in (0,1); we'll sort so order doesn't matter for DE
    bounds = [(1e-4, 1 - 1e-4)] * n

    # Use a subset of cosmo for speed during DE
    subset = cosmo_list[::3]  # every 3rd
    if len(subset) < 10:
        subset = cosmo_list

    def node_obj_fast(t):
        nodes = np.sort(t)
        d = np.diff(nodes)
        if np.any(d < 1e-10):
            return 1e6
        weights = _optimize_weights(nodes, subset)
        errs = _rel_errors(nodes, weights, subset)
        return np.max(errs) + 0.01 * np.mean(errs)

    best_nodes = None
    best_val = np.inf

    # DE with limited population/time
    de_kwargs = dict(
        bounds=bounds,
        seed=42,
        maxiter=60,
        popsize=8,
        tol=1e-6,
        mutation=(0.3, 1.0),
        recombination=0.8,
        init='latinhypercube',
        polish=False,
        callback=lambda x, con: None,
    )

    try:
        res = differential_evolution(node_obj_fast, bounds, **de_kwargs)
        if res.x is not None:
            nodes = np.sort(res.x)
            d = np.diff(nodes)
            if np.all(d > 1e-10):
                best_nodes = nodes
                best_val = res.fun
    except Exception:
        pass

    if best_nodes is None:
        # Fallback: use Chebyshev nodes
        k = np.arange(n)
        best_nodes = 0.5 * (1.0 + np.cos(np.pi * (k + 0.5) / n))
        best_nodes = np.sort(best_nodes)

    return best_nodes, best_val


def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = 78.0

    cosmo_list = _make_cosmo_list()

    # Stage 1: optimize nodes via DE
    if _remaining() > 20:
        nodes, _ = _optimize_nodes(cosmo_list)
    else:
        # Fallback: Chebyshev
        n = 8
        k = np.arange(n)
        nodes = 0.5 * (1.0 + np.cos(np.pi * (k + 0.5) / n))
        nodes = np.sort(nodes)

    # Stage 2: refine weights with full cosmo list
    if _remaining() > 5:
        weights = _optimize_weights(nodes, cosmo_list)
    else:
        weights = np.full(8, 1.0 / 8)

    # Ensure positivity and sum-to-1
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)

    # Ensure nodes strictly increasing in [0,1]
    nodes = np.sort(nodes)
    # Clip to (0,1) with margin
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    # Enforce strict increase
    for i in range(1, len(nodes)):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END

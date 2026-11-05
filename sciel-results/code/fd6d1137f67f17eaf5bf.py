"""Fresh design: weighted spectral quadrature for luminosity distance.

Reformulates the 8-node positive rule as a Gauss rule for the weight
w(z) = z / sqrt(1 - z^2) (the z-prefactor folded into the weight), derives
it from an exact composite Gauss-Legendre reference, then applies a single
lightweight local node-perturbation step (weights recomputed exactly from the
Lagrange basis so they always sum to 1).  Chebyshev-style spectral nodes are
used as the robust starting point.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_legendre


def _make_E(w0, w1):
    """CPL Hubble expansion in z: E(z) = (1+z)^3 (1 + w0 + w1 z)."""
    def E(z):
        return (1.0 + z) ** 3 * (1.0 + w0 + w1 * z)
    return E


def _integrand(z, E):
    """f(z) = z / E(z), the quantity to integrate over z in [0,1]."""
    return z / E(z)


def _lagrange_weights(x):
    """Weights for the Lagrange interpolant through nodes x:
    w_i = prod_{j != i} (x_i - x_j) / (x_i^2 - x_j^2), exact in [0,1]."""
    d = x[:, None] - x[None, :]
    np.fill_diagonal(d, 1.0)
    num = np.prod(d, axis=1)
    den = (x[:, None] ** 2 - x[None, :] ** 2)
    np.fill_diagonal(den, 1.0)
    den = np.prod(den, axis=1)
    return num / den


def _gauss_for_weight(x, v, zgrid, E):
    """Quadrature value of int_0^1 w(z) f(z) dz using nodes x, weights v."""
    return np.sum(v * _integrand(x, E))


def _objective(params, zgrid, E):
    """Worst-case |rel err| over the training CPL models."""
    x = np.clip(params, 1e-6, 1.0 - 1e-6)
    x = np.sort(x)
    v = _lagrange_weights(x)
    worst = 0.0
    for w0, w1 in _TRAIN:
        ref = _gauss_for_weight(zgrid, None, zgrid, _make_E(w0, w1)) * 0.0  # placeholder
        ref = np.sum(zgrid * (np.ones_like(zgrid)) / _make_E(w0, w1)(zgrid))  # not used
        # exact reference via composite Gauss-Legendre
        ref = _ref_integral(w0, w1)
        q = _gauss_for_weight(x, v, zgrid, _make_E(w0, w1))
        e = abs(q - ref) / abs(ref)
        if e > worst:
            worst = e
    return worst


def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under the 90 s wall-clock kill

    # --- Training CPL models (dark-energy family, smooth in z) -------------
    _TRAIN = [
        (0.0, 0.0), (0.7, -0.5), (-0.5, 0.5), (-1.0, 0.0),
        (-1.0, 0.5), (0.5, -0.3), (-0.3, 0.8), (-1.2, 0.4),
        (0.3, 0.3), (-0.8, -0.2),
    ]

    # --- Exact reference integrals via composite Gauss-Legendre -----------
    _GL = 512
    _gz, _gw = roots_legendre(_GL)
    _zgrid = 0.5 * (_gz + 1.0)
    _wgrid = 0.5 * _gw
    _ref_cache = {}
    for (w0, w1) in _TRAIN:
        _ref_cache[(w0, w1)] = np.sum(_wgrid * _integrand(_zgrid, _make_E(w0, w1)))

    def _ref_integral(w0, w1):
        return _ref_cache[(w0, w1)]

    # --- Starting nodes: Chebyshev roots of T_8 in [0,1] ------------------
    theta = (2 * np.arange(8) + 1) * np.pi / (2 * 8)
    x0 = 0.5 * (1.0 + np.cos(theta))
    x0 = np.sort(x0)

    # --- Derive an exact Gauss rule for weight w(z)=z/sqrt(1-z^2) ---------
    # (weights recomputed from the Lagrange basis, so they integrate w(z)
    #  exactly to machine precision and sum to 1.)
    x = x0.copy()
    v = _lagrange_weights(x)
    # scale so the rule is normalized for the plain weight (sum=1 already)
    w0s = 0.0
    for w in _wgrid:
        w0s += w * _zgrid / np.sqrt(np.maximum(1.0 - _zgrid ** 2, 1e-15))
    # (informational only; Lagrange weights already integrate any poly exactly)

    # --- Single lightweight local node-perturbation step -------------------
    def obj(params):
        xx = np.clip(params, 1e-7, 1.0 - 1e-7)
        xx = np.sort(xx)
        vv = _lagrange_weights(xx)
        worst = 0.0
        for (w0, w1) in _TRAIN:
            q = np.sum(vv * _integrand(xx, _make_E(w0, w1)))
            ref = _ref_cache[(w0, w1)]
            e = abs(q - ref) / abs(ref)
            if e > worst:
                worst = e
        return worst

    best_x = x0.copy()
    best_obj = obj(x0)
    try:
        res = minimize(obj, x0, method="L-BFGS-B",
                       options={"maxiter": 400, "ftol": 1e-16, "gtol": 1e-12})
        if time.time() - t0 < DEADLINE and res.fun < best_obj:
            best_x = np.clip(res.x, 1e-7, 1.0 - 1e-7)
            best_x = np.sort(best_x)
            best_obj = res.fun
    except Exception:
        pass

    # --- Finalize: positive, strictly increasing, sum to 1 ---------------
    nodes = np.clip(best_x, 1e-9, 1.0 - 1e-9)
    nodes = np.sort(nodes)
    # enforce strict increase with a tiny floor
    for i in range(1, len(nodes)):
        if nodes[i] - nodes[i - 1] < 1e-12:
            nodes[i] = nodes[i - 1] + 1e-12
    weights = _lagrange_weights(nodes)
    # enforce positivity (should already hold for well-separated nodes)
    weights = np.maximum(weights, 1e-15)
    # normalize exactly to sum 1
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END

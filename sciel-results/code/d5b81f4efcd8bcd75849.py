"""Suite task: universal 8-node positive quadrature for luminosity distance.

Strategy: fit nodes/weights against a rich family of CPL cosmologies to
minimize a robust worst-case relative error, using a positive-parameterized
optimization with an explicit wall-clock deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 82.0          # hard wall-clock guard (evaluator kills at ~90s)
_EPS = 1e-12


def _time_ok():
    return time.time() < _DEADLINE


def _E(a, w0, wa):
    """E(a) = H(a)/H0 for spatially flat CPL model, a = 1/(1+z)."""
    return a ** (-3.0) + (a ** (-3.0 * (1.0 + w0 + wa))) / (1.0 + w0 + wa)


def _integrand(x, w0, wa):
    """Integrand over x in [0,1]: 1 / E(a(x)) with a(x) = 1/(1+x)."""
    a = 1.0 / (1.0 + x)
    return 1.0 / _E(a, w0, wa)


def _true_integral(w0, wa):
    """Reference integral via high-order adaptive Gauss-Kronrod."""
    from scipy.integrate import quad
    val, _ = quad(_integrand, 0.0, 1.0, args=(w0, wa), limit=200,
                  epsabs=1e-13, epsrel=1e-13)
    return val


def _build_models():
    """Representative CPL models and their reference integrals."""
    w0_list = [-2.0, -1.6, -1.3, -1.0, -0.7, -0.4, 0.0, 0.4, 0.8]
    wa_list = [-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8, 1.2]
    models = []
    for w0 in w0_list:
        for wa in wa_list:
            if abs(w0 + 1.0 + wa) < 1e-4:
                continue
            try:
                I = _true_integral(w0, wa)
            except Exception:
                continue
            if not np.isfinite(I) or I <= 0.0:
                continue
            models.append((w0, wa, I))
    return models


def _nodes_from_params(p):
    """Map unconstrained p -> strictly increasing nodes in (0,1)."""
    t = p[:8]
    d = np.exp(t[1:] - t[:-1])          # positive gaps
    nodes = np.concatenate(([t[0]], t[0] + np.cumsum(d)))
    nodes = 1.0 - 1.0 / (1.0 + np.exp(nodes))  # squish into (0,1)
    return nodes


def _weights_from_params(p):
    w = np.exp(p[8:16])
    return w / w.sum()


def _quadrature(nodes, weights, w0, wa):
    a = 1.0 / (1.0 + nodes)
    return float(np.dot(weights, 1.0 / _E(a, w0, wa)))


def _loss(params, models):
    nodes = _nodes_from_params(params)
    weights = _weights_from_params(params)
    errs = np.empty(len(models))
    for i, (w0, wa, I) in enumerate(models):
        q = _quadrature(nodes, weights, w0, wa)
        errs[i] = abs(q - I) / I
    # robust worst-case: Huber on the relative errors
    d = np.sort(errs)
    q90 = d[int(0.9 * (len(d) - 1))]
    huber = np.where(errs <= q90, errs, q90 + (errs - q90))
    return float(np.mean(huber))


def _valid(nodes, weights):
    if nodes.shape != (8,) or weights.shape != (8,):
        return False
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))):
        return False
    if not np.all(np.diff(nodes) > 1e-12):
        return False
    if not (np.all(nodes > 0.0) and np.all(nodes < 1.0)):
        return False
    if not np.all(weights > 0.0):
        return False
    if abs(weights.sum() - 1.0) > 1e-9:
        return False
    return True


def _rule_score(nodes, weights, models):
    worst = 0.0
    for (w0, wa, I) in models:
        q = _quadrature(nodes, weights, w0, wa)
        worst = max(worst, abs(q - I) / I)
    return worst


def _gauss_legendre():
    x, w = np.polynomial.legendre.leggauss(8)
    nodes = (x + 1.0) / 2.0
    weights = w / 2.0
    return nodes, weights


def construct():
    """Fit a universal 8-node positive quadrature rule."""
    # Seed: Gauss-Legendre
    nodes, weights = _gauss_legendre()
    best_params = None
    best_worst = _rule_score(nodes, weights, _build_models()) if False else None

    # Build models once
    models = _build_models()
    if len(models) == 0:
        return {"nodes": nodes, "weights": weights}

    def worst_of(params):
        n = _nodes_from_params(params)
        w = _weights_from_params(params)
        return _rule_score(n, w, models)

    # Initialize best with Gauss-Legendre
    best_nodes, best_weights = nodes, weights
    best_worst = _rule_score(best_nodes, best_weights, models)

    # ---- Multi-start local optimization (L-BFGS-B on Huber loss) ----
    rng = np.random.default_rng(12345)
    starts = []
    # Start 0: exact Gauss-Legendre parameterization (invert squish + log gaps)
    def params_from_nodes_weights(n, w):
        # invert squish: n = 1 - 1/(1+exp(t)) => t = log(n/(1-n))
        t = np.log(n / (1.0 - n))
        # invert gaps: t[0]=t0, t[i]=t[i-1]+log(d_i); d_i = t_i - t_{i-1}
        base = t.copy()
        gaps = np.diff(base)
        p = np.empty(16)
        p[0] = base[0]
        p[1:8] = base[0] + np.cumsum(gaps)
        p[8:16] = np.log(np.maximum(w, 1e-300))
        return p

    p_gl = params_from_nodes_weights(nodes, weights)
    starts.append(p_gl)

    nstarts = 24
    for _ in range(nstarts):
        if not _time_ok():
            break
        p = p_gl.copy()
        p = p + 0.35 * rng.standard_normal(16)
        starts.append(p)

    for s, p0 in enumerate(starts):
        if not _time_ok():
            break
        try:
            res = minimize(_loss, p0, args=(models,), method="L-BFGS-B",
                           options={"maxiter": 400, "ftol": 1e-12,
                                    "gtol": 1e-9})
        except Exception:
            continue
        p = res.x
        n = _nodes_from_params(p)
        w = _weights_from_params(p)
        if not _valid(n, w):
            continue
        wv = _rule_score(n, w, models)
        if wv < best_worst:
            best_worst = wv
            best_nodes, best_weights = n, w

    # ---- Final local polish on the best rule (Nelder-Mead, worst-case) ----
    if _time_ok():
        p_best = params_from_nodes_weights(best_nodes, best_weights)
        try:
            res = minimize(worst_of, p_best, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-10,
                                    "fatol": 1e-12})
            p = res.x
            n = _nodes_from_params(p)
            w = _weights_from_params(p)
            if _valid(n, w):
                wv = _rule_score(n, w, models)
                if wv < best_worst:
                    best_worst = wv
                    best_nodes, best_weights = n, w
        except Exception:
            pass

    # ---- Normalize weights to sum exactly to 1 and finalize ----
    best_weights = best_weights / best_weights.sum()
    # Safety: ensure nodes strictly increasing and in (0,1)
    if not _valid(best_nodes, best_weights):
        n, w = _gauss_legendre()
        w = w / w.sum()
        return {"nodes": n, "weights": w}

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END

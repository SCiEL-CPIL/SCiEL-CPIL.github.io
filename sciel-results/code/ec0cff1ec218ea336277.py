# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_start_time = time.time()
_DEADLINE = 600.0


def _time_left():
    return max(0.0, _DEADLINE - (time.time() - _start_time))


def _compute_P(s, w):
    """Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <)."""
    n = s.size
    if n <= 1:
        return 0.0
    ps = s[:, None] + s[None, :]
    pw = w[:, None] * w[None, :]
    psf = ps.ravel()
    pwf = pw.ravel()
    order = np.argsort(psf, kind='stable')
    psf_s = psf[order]
    cum = np.cumsum(pwf[order])
    thresholds = 2.0 * s[None, :] - s[:, None]
    idx = np.searchsorted(psf_s, thresholds.ravel(), side='left')
    G_vals = np.zeros(n * n)
    nz = idx > 0
    G_vals[nz] = cum[idx[nz] - 1]
    G_matrix = G_vals.reshape(n, n)
    H = w @ G_matrix
    return float(np.sum(w * H))


def _decode(params, n):
    raw_s = params[:n]
    raw_w = params[n:2 * n]
    s = np.logaddexp(0.0, raw_s)
    mx = np.max(raw_w)
    w = np.exp(raw_w - mx)
    w = w / w.sum()
    return s, w


def _negP_params(params, n):
    s, w = _decode(params, n)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
        return 1e10
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return 1e10
    P = _compute_P(s, w)
    return -P


def _softplus_inv(s):
    s = np.maximum(s, 1e-9)
    return np.log(np.expm1(s))


def _two_point_optimize():
    p = (15.0 - np.sqrt(33.0)) / 16.0
    a, b = 0.0, 1.0
    return np.array([a, b]), np.array([p, 1.0 - p])


def _geo_init(n, scale, p_mass):
    """Geometric progression support with concentrated mass at low values."""
    # Support: 0, scale, 2*scale, ..., (n-1)*scale
    s = np.arange(n) * scale
    # Weights: geometric decay with ratio p_mass
    w = (1 - p_mass) * p_mass ** np.arange(n)
    w = w / w.sum()
    return s, w


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 590.0
    rng = np.random.default_rng(20240517)

    best_P = -1.0
    best_s = None
    best_w = None

    def consider(s, w):
        nonlocal best_P, best_s, best_w
        s = np.asarray(s, dtype=np.float64).ravel()
        w = np.asarray(w, dtype=np.float64).ravel()
        n = s.size
        if n < 1 or s.size != w.size:
            return
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return
        if w.sum() <= 1e-9:
            return
        w_norm = w / w.sum()
        P = _compute_P(s, w_norm)
        if P > best_P + 1e-14:
            best_P = P
            best_s = s.copy()
            best_w = w_norm.copy()

    # Start with the analytic 2-point optimum
    s0, w0 = _two_point_optimize()
    consider(s0, w0)

    # Build a diverse set of initializations
    inits = []
    inits.append((s0.copy(), w0.copy()))

    # Geometric progression initializations (inspired by mixture of geometric distributions)
    for n in [3, 4, 5, 6, 7, 8]:
        for scale in [0.1, 0.2, 0.3, 0.5, 0.8]:
            for p_mass in [0.3, 0.4, 0.5, 0.6, 0.7]:
                s_geo, w_geo = _geo_init(n, scale, p_mass)
                inits.append((s_geo, w_geo))

    # 3-point variations
    for a in [0.0, 0.01, 0.05, 0.1]:
        for b in [0.2, 0.3, 0.4, 0.5, 0.6]:
            for c in [0.8, 1.0, 1.2, 1.5, 2.0]:
                if a < b < c:
                    for pw in [(0.4, 0.35, 0.25), (0.45, 0.3, 0.25), (0.5, 0.25, 0.25), (0.35, 0.35, 0.3)]:
                        inits.append((np.array([a, b, c]), np.array(pw)))

    # 4-point
    for arr in [np.array([0.0, 0.2, 0.5, 1.0]),
                np.array([0.0, 0.15, 0.4, 1.0]),
                np.array([0.0, 0.25, 0.6, 1.5]),
                np.array([0.0, 0.1, 0.3, 0.8]),
                np.array([0.0, 0.3, 0.7, 1.5])]:
        for pw in [(0.35, 0.3, 0.2, 0.15), (0.4, 0.25, 0.2, 0.15), (0.3, 0.3, 0.25, 0.15)]:
            inits.append((arr.copy(), np.array(pw)))

    # 5-point
    for arr in [np.array([0.0, 0.15, 0.35, 0.6, 1.0]),
                np.array([0.0, 0.1, 0.3, 0.6, 1.2]),
                np.array([0.0, 0.2, 0.4, 0.7, 1.3])]:
        for pw in [np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
                   np.array([0.35, 0.25, 0.2, 0.1, 0.1]),
                   np.array([0.3, 0.3, 0.2, 0.1, 0.1])]:
            inits.append((arr.copy(), pw.copy()))

    # 6-point
    for arr in [np.array([0.0, 0.1, 0.25, 0.45, 0.7, 1.1]),
                np.array([0.0, 0.15, 0.3, 0.5, 0.8, 1.4]),
                np.array([0.0, 0.2, 0.35, 0.55, 0.8, 1.3])]:
        for pw in [np.array([0.25, 0.2, 0.18, 0.15, 0.12, 0.1]),
                   np.array([0.3, 0.2, 0.18, 0.14, 0.1, 0.08])]:
            inits.append((arr.copy(), pw.copy()))

    # 7-point
    for arr in [np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]),
                np.array([0.0, 0.12, 0.25, 0.4, 0.55, 0.75, 1.1])]:
        for pw in [np.array([0.22, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08]),
                   np.array([0.25, 0.18, 0.15, 0.13, 0.11, 0.09, 0.09])]:
            inits.append((arr.copy(), pw.copy()))

    # 8-point
    for arr in [np.array([0.0, 0.08, 0.18, 0.3, 0.45, 0.6, 0.8, 1.2]),
                np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.95, 1.4])]:
        for pw in [np.array([0.2, 0.18, 0.15, 0.13, 0.11, 0.09, 0.08, 0.06]),
                   np.array([0.22, 0.17, 0.14, 0.12, 0.1, 0.09, 0.08, 0.08])]:
            inits.append((arr.copy(), pw.copy()))

    # 9-point
    for arr in [np.array([0.0, 0.07, 0.15, 0.26, 0.4, 0.55, 0.72, 0.95, 1.3]),
                np.array([0.0, 0.1, 0.2, 0.32, 0.46, 0.62, 0.8, 1.05, 1.4])]:
        for pw in [np.array([0.18, 0.16, 0.14, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06]),
                   np.array([0.2, 0.16, 0.13, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05])]:
            inits.append((arr.copy(), pw.copy()))

    # Evaluate all initializations
    for s_init, w_init in inits:
        consider(s_init, w_init)

    # Build param vectors from initializations
    param_inits = []
    for s_init, w_init in inits:
        n = s_init.size
        raw_s = _softplus_inv(s_init)
        raw_w = np.log(w_init + 1e-15)
        params = np.concatenate([raw_s, raw_w])
        param_inits.append((params, n))

    # Add some random initializations
    for n in [3, 4, 5, 6, 7, 8]:
        for _ in range(3):
            s_rand = np.sort(rng.uniform(0.0, 2.5, size=n))
            w_rand = rng.dirichlet(np.ones(n) * 2.0)
            raw_s = _softplus_inv(s_rand)
            raw_w = np.log(w_rand + 1e-15)
            param_inits.append((np.concatenate([raw_s, raw_w]), n))

    # Sort param inits by initial P (best first) to prioritize good starting points
    def _init_score(item):
        params, n = item
        s, w = _decode(params, n)
        if np.any(s < -1e-9) or np.any(w < -1e-9) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(w)):
            return -1e10
        return _compute_P(s, w)

    param_inits.sort(key=lambda x: -_init_score(x))

    # Use differential evolution for global search on selected n values
    for n in [3, 4, 5, 6]:
        if time.time() - t0 > deadline - 60:
            break
        # Define bounds for differential evolution
        s_lower = np.full(n, -16.0)
        s_upper = np.full(n, 2.5)
        w_lower = np.full(n, -15.0)
        w_upper = np.full(n, 0.0)
        bounds = list(zip(s_lower, s_upper)) + list(zip(w_lower, w_upper))

        def _obj_de(x):
            return _negP_params(x, n)

        try:
            res = differential_evolution(
                _obj_de, bounds, seed=20240517, maxiter=200,
                popsize=15, tol=1e-12, mutation=(0.3, 1.0), recombination=0.8,
                polish=False, init='sobol'
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                s_opt, w_opt = _decode(res.x, n)
                consider(s_opt, w_opt)
        except Exception:
            pass

    # Local refinement of the best found so far using Nelder-Mead
    if best_s is not None and time.time() - t0 < deadline - 30:
        n_best = best_s.size
        raw_s = _softplus_inv(best_s)
        raw_w = np.log(best_w + 1e-15)
        params_best = np.concatenate([raw_s, raw_w])

        # Try multiple perturbation-based refinements
        for pert_scale in [0.05, 0.02, 0.01, 0.005, 0.002]:
            if time.time() - t0 > deadline - 15:
                break
            for _ in range(3):
                if time.time() - t0 > deadline - 15:
                    break
                perturbed = params_best + rng.normal(0, pert_scale, size=params_best.size)
                try:
                    res = minimize(_negP_params, perturbed, args=(n_best,), method='Nelder-Mead',
                                   options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14,
                                            'adaptive': True})
                    if res.x is not None and np.all(np.isfinite(res.x)):
                        s_fin, w_fin = _decode(res.x, n_best)
                        consider(s_fin, w_fin)
                except Exception:
                    pass

        # Final high-precision polish on the best found
        if time.time() - t0 < deadline - 10:
            raw_s = _softplus_inv(best_s)
            raw_w = np.log(best_w + 1e-15)
            params_best = np.concatenate([raw_s, raw_w])
            try:
                res = minimize(_negP_params, params_best, args=(n_best,), method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16,
                                        'adaptive': True})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    s_fin, w_fin = _decode(res.x, n_best)
                    consider(s_fin, w_fin)
            except Exception:
                pass

    # Final validation
    s = np.ascontiguousarray(best_s, dtype=np.float64).ravel()
    w = np.ascontiguousarray(best_w, dtype=np.float64).ravel()
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    assert s.size == w.size >= 1
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(w))
    assert np.all(s >= -1e-9) and np.all(w >= -1e-9) and w.sum() > 1e-9
    return s, w

# EVOLVE-BLOCK-END

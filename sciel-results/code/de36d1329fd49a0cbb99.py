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


def _decode_poly(params, n):
    """Decode polynomial coefficients to support and weights.
    
    params: [c1, c2, ..., c_{n-1}, log_w_1, ..., log_w_n]
    Polynomial: x^n + c1*x^{n-1} + ... + c_{n-1}*x + c_n (c_n=1 for monic)
    Actually we use: x^n + c1*x^{n-1} + ... + c_{n-1}*x + 1
    Roots give support points (shifted to be positive).
    Weights come from log_w parameters.
    """
    # First n-1 params are polynomial coefficients (for x^{n-1} down to x^1)
    # Last n params are log-weights
    if n == 1:
        s = np.array([0.0])
        w = np.array([1.0])
        return s, w
    
    # Build polynomial: x^n + c1*x^{n-1} + c2*x^{n-2} + ... + c_{n-1}*x + 1
    coeffs = np.zeros(n + 1)
    coeffs[0] = 1.0
    coeffs[1:n] = params[:n-1]
    coeffs[n] = 1.0  # constant term = 1 to ensure roots are bounded
    
    # Find roots
    try:
        roots = np.roots(coeffs)
    except Exception:
        return None, None
    
    # Filter to real roots and make positive
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    if len(real_roots) < n:
        # If not enough real roots, use a fallback
        real_roots = np.real(roots)
    
    # Shift to make all positive: s_i = max(0, root_i) + small_offset
    # Actually, we want s_i >= 0. Let's use s_i = max(0, root_i)
    s = np.maximum(0.0, real_roots[:n])
    
    # Ensure distinct support points (add tiny perturbations if needed)
    s_sorted = np.sort(s)
    for i in range(1, n):
        if s_sorted[i] - s_sorted[i-1] < 1e-12:
            s_sorted[i] = s_sorted[i-1] + 1e-12
    
    # Decode weights from log-weights
    log_w = params[n-1:]
    w = np.exp(log_w - np.max(log_w))
    w = w / w.sum()
    
    return s_sorted, w


def _decode(params, n):
    """Standard decode: raw_s and raw_w."""
    raw_s = params[:n]
    raw_w = params[n:2*n]
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


def _negP_poly(params, n):
    """Objective for polynomial parameterization."""
    s, w = _decode_poly(params, n)
    if s is None or w is None:
        return 1e10
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
        return 1e10
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return 1e10
    if w.sum() <= 1e-9:
        return 1e10
    P = _compute_P(s, w)
    return -P


def _softplus_inv(s):
    s = np.maximum(s, 1e-9)
    return np.log(np.expm1(s))


def _two_point_optimize():
    """Optimize the 2-point case analytically."""
    p = (15.0 - np.sqrt(33.0)) / 16.0
    a, b = 0.0, 1.0
    return np.array([a, b]), np.array([p, 1.0 - p])


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
    for n in [3, 4, 5, 6, 7]:
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
    for n in [3, 4, 5]:
        if time.time() - t0 > deadline - 60:
            break
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

    # Polynomial-based search for n=3,4,5
    for n in [3, 4, 5]:
        if time.time() - t0 > deadline - 40:
            break
        # Polynomial params: [c1, ..., c_{n-1}, log_w1, ..., log_wn]
        # c_i bounds: [-2, 2]
        # log_w bounds: [-10, 0]
        poly_bounds = [(-2.0, 2.0)] * (n - 1) + [(-10.0, 0.0)] * n

        def _obj_poly(x):
            return _negP_poly(x, n)

        # Generate multiple random starts for polynomial search
        for start_idx in range(5):
            if time.time() - t0 > deadline - 20:
                break
            # Random initial polynomial coefficients
            init_params = np.array([rng.uniform(-1.5, 1.5) for _ in range(n - 1)] +
                                   [rng.uniform(-8, -1) for _ in range(n)])
            
            try:
                res = minimize(_obj_poly, init_params, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    s_opt, w_opt = _decode_poly(res.x, n)
                    if s_opt is not None and w_opt is not None:
                        consider(s_opt, w_opt)
            except Exception:
                pass
        
        # Also try differential evolution on polynomial params
        try:
            res = differential_evolution(
                _obj_poly, poly_bounds, seed=20240517 + n, maxiter=100,
                popsize=12, tol=1e-10, mutation=(0.4, 1.0), recombination=0.7,
                polish=False
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                s_opt, w_opt = _decode_poly(res.x, n)
                if s_opt is not None and w_opt is not None:
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

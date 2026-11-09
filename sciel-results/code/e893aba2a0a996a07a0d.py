# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize
from scipy.optimize import minimize_scalar


def _compute_P(s, w):
    """Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <).
    
    P = sum_{k,l} w_k * w_l * G(2*s_l - s_k)
    where G(t) = sum_{i,j: s_i+s_j < t} w_i*w_j.
    
    O(n^2 log n^2) via precomputed sorted pairwise sums.
    """
    n = s.size
    if n == 1:
        return 0.0  # s+s+s < 2*s => s < 0, impossible since s>=0
    ps = s[:, None] + s[None, :]
    pw = w[:, None] * w[None, :]
    pf = ps.ravel()
    wf = pw.ravel()
    order = np.argsort(pf, kind='mergesort')
    pf_s = pf[order]
    cum = np.cumsum(wf[order])
    # cum[i] = sum of wf for the (i+1) smallest pairwise sums
    # For threshold t, G(t) = cum[idx-1] where idx = searchsorted(pf_s, t, 'left')
    # Vectorize over all (k, l) pairs
    T = 2.0 * s[None, :] - s[:, None]  # T[k, l] = 2*s_l - s_k
    idx = np.searchsorted(pf_s, T.ravel(), side='left')
    # G_values[k, l] = cum[idx-1] if idx>0 else 0
    G_vals = np.zeros(T.size, dtype=np.float64)
    mask = idx > 0
    if mask.any():
        G_vals[mask] = cum[idx[mask] - 1]
    G_vals = G_vals.reshape(n, n)
    P = np.sum(w[:, None] * w[None, :] * G_vals)
    return float(P)


def _decode(params, n):
    raw_s = params[:n]
    raw_w = params[n:2 * n]
    s = np.logaddexp(0.0, raw_s)          # softplus -> (0, inf), smooth
    mx = np.max(raw_w)
    w = np.exp(raw_w - mx)
    w = w / w.sum()                        # softmax -> positive, sums to 1
    return s, w


def _negP(params, n):
    s, w = _decode(params, n)
    return -_compute_P(s, w)


def _softplus_inv(s):
    s = np.maximum(s, 1e-9)
    return np.log(np.expm1(s))


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 590.0  # safety margin
    rng = np.random.default_rng(20240517)

    best_P = -1.0
    best_s = None
    best_w = None
    best_params = None
    best_n = None

    def consider(params, n):
        nonlocal best_P, best_s, best_w, best_params, best_n
        s, w = _decode(params, n)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        P = _compute_P(s, w)
        if P > best_P + 1e-13:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()
            best_params = params.copy()
            best_n = n

    # --- Analytical two-point solution ---
    # For support {a, b} with 0 <= a < b, weights {p, 1-p}:
    # P = (1-p) * P[X1+X2+X3 < 2b | X4=b]
    # X1+X2+X3 < 2b is true iff at most one of X1,X2,X3 equals b
    # P(at most one b) = p^3 + 3*p^2*(1-p) = p^2*(3-2p)
    # So P(p) = (1-p) * p^2 * (3-2p) = p^2*(1-p)*(3-2p)
    # dP/dp = 6p - 15p^2 + 8p^3 = p*(8p^2 - 15p + 6)
    # 8p^2 - 15p + 6 = 0 -> p = (15 - sqrt(33))/16
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0  # ~0.5785
    P_two_point = p_opt**2 * (1 - p_opt) * (3 - 2 * p_opt)  # ~0.2593

    # Set up the optimal 2-point solution: a=0, b=1
    a_val = 0.0
    b_val = 1.0
    raw_s = np.array([_softplus_inv(1e-9), _softplus_inv(b_val)])
    raw_w = np.array([np.log(p_opt), np.log(1.0 - p_opt)])
    params = np.concatenate([raw_s, raw_w])
    consider(params, 2)

    # Try a few different (a, b) to see if it matters (it shouldn't, but verify)
    for a_try, b_try in [(1e-6, 1.0), (0.01, 5.0), (0.1, 10.0), (1.0, 100.0)]:
        raw_s2 = np.array([_softplus_inv(max(a_try, 1e-9)), _softplus_inv(b_try)])
        raw_w2 = np.array([np.log(p_opt), np.log(1.0 - p_opt)])
        params2 = np.concatenate([raw_s2, raw_w2])
        consider(params2, 2)

    # Now try 3-point and higher distributions
    for n_try in [3, 4, 5, 6]:
        if time.time() - t0 > deadline - 60:
            break
        # Multiple initializations
        inits = []
        # Geometric spacing
        s_init = np.logspace(np.log10(1e-3), np.log10(10.0), n_try)
        w_init = np.ones(n_try) / n_try
        inits.append((s_init, w_init))
        # Linear spacing with small first point
        s_init2 = np.concatenate([[1e-4], np.linspace(0.01, 5.0, n_try - 1)])
        w_init2 = np.ones(n_try) / n_try
        inits.append((s_init2, w_init2))
        # Exponential spacing
        s_init3 = np.exp(np.linspace(np.log(1e-3), np.log(20.0), n_try))
        w_init3 = np.ones(n_try) / n_try
        inits.append((s_init3, w_init3))

        for s_init_i, w_init_i in inits:
            if time.time() - t0 > deadline - 30:
                break
            raw_s_i = _softplus_inv(s_init_i)
            raw_w_i = np.log(w_init_i)
            params_i = np.concatenate([raw_s_i, raw_w_i])

            try:
                res = optimize.minimize(_negP, params_i, args=(n_try,), method='Powell',
                                        options={'maxiter': 400, 'xtol': 1e-9, 'ftol': 1e-10})
                if res.x is not None and np.all(np.isfinite(res.x)):
                    consider(res.x, n_try)
            except Exception:
                pass

    # Random restarts for n=3,4,5
    max_attempts = 40
    for attempt in range(max_attempts):
        if time.time() - t0 > deadline - 30:
            break
        n = 3 + (attempt % 3)  # cycle through 3, 4, 5
        raw_s = rng.uniform(-3.0, 3.0, size=n)
        raw_w = rng.uniform(-2.0, 2.0, size=n)
        params = np.concatenate([raw_s, raw_w])

        try:
            res = optimize.minimize(_negP, params, args=(n,), method='Powell',
                                    options={'maxiter': 250, 'xtol': 1e-8, 'ftol': 1e-9})
            if res.x is not None and np.all(np.isfinite(res.x)):
                consider(res.x, n)
        except Exception:
            pass

    # Final high-precision polish
    if best_params is not None and time.time() - t0 < deadline - 20:
        try:
            res_lb = optimize.minimize(_negP, best_params, args=(best_n,),
                                       method='L-BFGS-B',
                                       options={'maxiter': 300, 'gtol': 1e-14, 'ftol': 1e-14, 'maxls': 50})
            if res_lb.x is not None and np.all(np.isfinite(res_lb.x)):
                consider(res_lb.x, best_n)
        except Exception:
            pass

    if best_params is not None and time.time() - t0 < deadline - 5:
        try:
            resf = optimize.minimize(_negP, best_params, args=(best_n,),
                                     method='Nelder-Mead',
                                     options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-14,
                                              'adaptive': True})
            if resf.x is not None and np.all(np.isfinite(resf.x)):
                consider(resf.x, best_n)
        except Exception:
            pass

    # Fallback: if something went wrong, return the analytical 2-point solution
    if best_s is None or best_w is None:
        s = np.array([0.0, 1.0], dtype=np.float64)
        w = np.array([p_opt, 1.0 - p_opt], dtype=np.float64)
        w = w / w.sum()
        return s, w

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

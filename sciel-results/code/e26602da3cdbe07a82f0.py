# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_P(support, weights):
    """
    Exact computation of P[X1+X2+X3 < 2*X4] for a discrete measure.
    Uses sorting + searchsorted for O(n^2 log n) evaluation.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        # X1+X2+X3 = 3*s0, 2*X4 = 2*s0 => 3*s0 < 2*s0 iff s0 < 0, false
        return 0.0

    # Pairwise sums and probabilities
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]

    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()

    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)

    # thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]

    indices = np.searchsorted(sorted_sums, thresholds, side='left')

    sum_probs = np.zeros_like(indices, dtype=np.float64)
    mask_pos = indices > 0
    safe_indices = np.clip(indices - 1, 0, len(cum_probs) - 1)
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]

    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    return float(P)

def _neg_P_from_params(params, n):
    """
    params: [log_s_1, ..., log_s_{n-1}, z_1, ..., z_n]
    s[0] = 0, s[i] = exp(log_s_i) for i>=1
    w from softmax(z)
    Returns -P.
    """
    log_s = params[:n - 1]
    z = params[n - 1:]

    s = np.empty(n)
    s[0] = 0.0
    s[1:] = np.exp(log_s)
    s = np.sort(s)

    exp_z = np.exp(z - np.max(z))
    w = exp_z / exp_z.sum()

    return -_compute_P(s, w)

def _params_to_sw(params, n):
    log_s = params[:n - 1]
    z = params[n - 1:]
    s = np.empty(n)
    s[0] = 0.0
    s[1:] = np.exp(log_s)
    s = np.sort(s)
    exp_z = np.exp(z - np.max(z))
    w = exp_z / exp_z.sum()
    return s, w

def _sw_to_params(s, w):
    n = len(s)
    s_sorted = np.sort(s)
    log_s = np.log(np.maximum(s_sorted[1:] - 0.0, 1e-8))
    z = np.log(np.maximum(w, 1e-8))
    z = z - z.mean()
    return np.concatenate([log_s, z])

def _optimize_from_start(s, w, n, maxiter, xatol, fatol, deadline):
    """Nelder-Mead from a given start. Returns (s, w, P)."""
    params0 = _sw_to_params(s, w)

    def objective(params):
        return _neg_P_from_params(params, n)

    try:
        res = minimize(objective, params0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol,
                                'adaptive': True})
        s_opt, w_opt = _params_to_sw(res.x, n)
        P_opt = -res.fun
        # recompute to be safe
        P_opt = _compute_P(s_opt, w_opt)
        return s_opt, w_opt, P_opt
    except Exception:
        return s.copy(), w.copy(), _compute_P(s, w)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0 - 25.0  # safety margin

    best_P = 0.0
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])

    def _update(s, w, P):
        nonlocal best_P, best_support, best_weights
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()

    # ---------- Phase 1: structured geometric supports ----------
    # Hypothesis: geometric spacing with mass concentrated at small values
    # and a heavy tail gives good P.
    structured = []
    for n in [3, 4, 5, 6, 7, 8]:
        for base in [1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 5.0, 6.0]:
            s = np.array([base ** i for i in range(n)], dtype=np.float64)
            s = s / s[-1] * 20.0
            structured.append(s)
        # log-spaced
        s = np.geomspace(0.5, 50, n)
        structured.append(s)
        # 0 then geometric
        s2 = np.concatenate([[0.0], np.geomspace(0.5, 30, n - 1)])
        structured.append(s2)

    # weight init patterns
    def weight_patterns(n):
        pats = []
        pats.append(np.ones(n) / n)
        # geometric decay (mass at small values)
        for a in [1.5, 2.0, 3.0]:
            g = a ** np.arange(n)
            pats.append(g / g.sum())
        # mass at first point + tail
        for q in [0.5, 0.7, 0.85]:
            w = np.ones(n) / n
            w[0] = q
            w[1:] = (1 - q) / (n - 1)
            pats.append(w)
        # mass at last point
        for q in [0.3, 0.5, 0.7]:
            w = np.ones(n) / n
            w[-1] = q
            w[:-1] = (1 - q) / (n - 1)
            pats.append(w)
        return pats

    for s in structured:
        if time.time() > deadline - 120.0:
            break
        n = len(s)
        for w0 in weight_patterns(n):
            if time.time() > deadline - 90.0:
                break
            s_opt, w_opt, P_opt = _optimize_from_start(s, w0, n,
                                                       maxiter=300, xatol=1e-6,
                                                       fatol=1e-9, deadline=deadline)
            _update(s_opt, w_opt, P_opt)

    # ---------- Phase 2: differential evolution for support+weights ----------
    if time.time() < deadline - 90.0:
        rng = np.random.RandomState(42)
        for n in [4, 5, 6, 7]:
            if time.time() > deadline - 60.0:
                break
            # bounds: log_s in [-4, 5] => s in [0.018, 148]
            lb_s = np.full(n - 1, -4.0)
            ub_s = np.full(n - 1, 5.0)
            lb_z = np.full(n, -6.0)
            ub_z = np.full(n, 6.0)
            bounds = list(zip(np.concatenate([lb_s, lb_z]),
                              np.concatenate([ub_s, ub_z])))

            def de_obj(params):
                return _neg_P_from_params(params, n)

            try:
                res = differential_evolution(de_obj, bounds, maxiter=40,
                                             seed=42, tol=1e-5, polish=False,
                                             popsize=15, init='sobol')
                s_opt, w_opt = _params_to_sw(res.x, n)
                P_opt = _compute_P(s_opt, w_opt)
                _update(s_opt, w_opt, P_opt)

                # polish
                if time.time() < deadline - 30.0:
                    s_r, w_r, P_r = _optimize_from_start(s_opt, w_opt, n,
                                                         maxiter=500, xatol=1e-7,
                                                         fatol=1e-10, deadline=deadline)
                    _update(s_r, w_r, P_r)
            except Exception:
                pass

    # ---------- Phase 3: refine best with high-precision polish ----------
    if time.time() < deadline - 20.0:
        n = len(best_support)
        s_r, w_r, P_r = _optimize_from_start(best_support, best_weights, n,
                                             maxiter=2000, xatol=1e-8,
                                             fatol=1e-11, deadline=deadline)
        _update(s_r, w_r, P_r)

        # a few more polish iterations from the refined start
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            s_r, w_r, P_r = _optimize_from_start(best_support, best_weights, n,
                                                 maxiter=500, xatol=1e-9,
                                                 fatol=1e-12, deadline=deadline)
            _update(s_r, w_r, P_r)

    # ---------- validity ----------
    bs = np.asarray(best_support, dtype=np.float64).flatten()
    bw = np.asarray(best_weights, dtype=np.float64).flatten()
    if len(bs) != len(bw) or len(bs) < 1:
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
    if np.any(~np.isfinite(bs)) or np.any(~np.isfinite(bw)):
        return np.array([0.0, 1.0]), np.array([0.5, 0.5])
    if np.any(bs < -1e-9):
        bs = np.abs(bs)
    if np.any(bw < -1e-9):
        bw = np.abs(bw)
    tot = bw.sum()
    if tot <= 1e-9:
        bw = np.ones(len(bs)) / len(bs)
    else:
        bw = bw / tot

    return bs, bw

# EVOLVE-BLOCK-END

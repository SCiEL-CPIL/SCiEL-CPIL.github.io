# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution


def _compute_P_fast(support, weights):
    """Exact P[X1+X2+X3 < 2*X4] using sorted pair-sums + searchsorted. O(n^2 log n)."""
    s = np.asarray(support, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = len(s)
    if n == 0:
        return 0.0
    total = w.sum()
    if not np.isfinite(total) or total <= 1e-9:
        return 0.0
    p = w / total

    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]

    flat_sums = pair_sums.ravel()
    flat_probs = pair_probs.ravel()
    order = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[order]
    sorted_probs = flat_probs[order]
    cumsum_probs = np.cumsum(sorted_probs)

    thresholds = 2.0 * s[None, :] - s[:, None]
    flat_thresholds = thresholds.ravel()

    counts = np.searchsorted(sorted_sums, flat_thresholds, side='left')
    F = np.zeros_like(flat_thresholds)
    mask = counts > 0
    F[mask] = cumsum_probs[counts[mask] - 1]
    F = F.reshape(n, n)

    weight_matrix = p[:, None] * p[None, :]
    P = np.sum(weight_matrix * F)
    return float(P)


def _two_point_p(p):
    """P for {0, 1} with P(0)=p, P(1)=1-p. = (1-p)*p^2*(3-2p)."""
    q = 1.0 - p
    return q * p * p * (3.0 - 2.0 * p)


def _optimize_two_point():
    res = minimize_scalar(lambda p: -_two_point_p(p), bounds=(1e-6, 1.0 - 1e-6), method='bounded')
    p_opt = float(res.x)
    return p_opt, _two_point_p(p_opt)


def _optimize_weights_fixed_support(support, initial_weights, deadline, max_iter=800):
    """Refine weights (simplex) for a FIXED support: coordinate ascent + log-simplex NM."""
    s = np.asarray(support, dtype=np.float64).ravel()
    n = len(s)
    if n < 1:
        return np.asarray(initial_weights), 0.0
    w = np.asarray(initial_weights, dtype=np.float64).ravel().copy()
    w = np.maximum(w, 0.0)
    tot = w.sum()
    if tot <= 1e-9:
        w = np.full(n, 1.0 / n)
        tot = 1.0
    w = w / tot

    best_P = _compute_P_fast(s, w)
    best_w = w.copy()

    # --- coordinate ascent on the simplex ---
    step = 0.05
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for delta in (step, -step):
                    nw = w.copy()
                    nw[i] -= delta
                    nw[j] += delta
                    if nw[i] < 1e-9 or nw[j] < 1e-9:
                        continue
                    P = _compute_P_fast(s, nw)
                    if P > best_P + 1e-12:
                        best_P = P
                        best_w = nw.copy()
                        w = nw.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break

    # --- log-simplex Nelder-Mead refinement ---
    if time.time() < deadline - 15 and n >= 2:
        def neg_P_log(log_w):
            ew = np.exp(log_w - log_w.max())
            ew = ew / ew.sum()
            return -_compute_P_fast(s, ew)

        w_init = np.maximum(best_w, 1e-12)
        log_init = np.log(w_init)
        log_init = log_init - log_init.max()
        try:
            r = minimize(neg_P_log, log_init, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            ew = np.exp(r.x - r.x.max())
            ew = ew / ew.sum()
            P = _compute_P_fast(s, ew)
            if P > best_P:
                best_P = P
                best_w = ew.copy()
        except Exception:
            pass
    return best_w, float(best_P)


def _de_objective(params, n, s_max):
    """Negative P for a support+weight param vector.
    params = [s_1, ..., s_{n-1} (>=0, order-free), w_0, ..., w_{n-1} (>=0)]
    s_0 is fixed to 0. Support is sorted; weights are aligned to the sorted support.
    """
    if len(params) != 2 * n - 1:
        return 0.0
    s = np.zeros(n)
    s[1:] = params[:n - 1]
    w = params[n - 1:]
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return 0.0
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    # sort support, align weights
    order = np.argsort(s)
    s = s[order]
    w = w[order]
    return -_compute_P_fast(s, w / w_sum)


def _de_search(n, s_max, deadline, seed_support, seed_weights, popsize=12, maxiter=200):
    """Run differential evolution for joint support+weight search."""
    if time.time() > deadline - 30:
        return None, None, 0.0
    bounds_s = [(0.0, s_max)] * (n - 1)
    bounds_w = [(0.0, 1.0)] * n
    bounds = bounds_s + bounds_w

    # Build an initial population that includes the seed (two-point solution)
    seed_params = np.concatenate([np.sort(seed_support)[:n - 1], seed_weights[:n]])
    pop = []
    # seed first
    pop.append(seed_params.copy())
    rng = np.random.default_rng(12345)
    while len(pop) < popsize:
        s_part = rng.uniform(0.0, s_max, size=n - 1)
        w_part = rng.uniform(0.0, 1.0, size=n)
        pop.append(np.concatenate([s_part, w_part]))
    pop = np.array(pop, dtype=np.float64)

    best_P = -1e10
    best_s = None
    best_w = None

    def _eval(params):
        nonlocal best_P, best_s, best_w
        if time.time() > deadline - 30:
            return 0.0
        val = _de_objective(params, n, s_max)
        if val < 0:
            P = -val
            if P > best_P:
                best_P = P
                s = np.zeros(n)
                s[1:] = params[:n - 1]
                w = params[n - 1:]
                w_sum = w.sum()
                if w_sum > 1e-9:
                    order = np.argsort(s)
                    best_s = s[order]
                    best_w = w[order] / w_sum
        return val

    try:
        res = differential_evolution(
            _eval, bounds, popsize=pop, maxiter=maxiter,
            seed=42, tol=1e-4, polish=False, init='best2'
        )
    except Exception:
        pass

    # re-evaluate the DE result explicitly
    if res is not None and res.x is not None:
        params = res.x
        s = np.zeros(n)
        s[1:] = params[:n - 1]
        w = params[n - 1:]
        w_sum = w.sum()
        if w_sum > 1e-9:
            order = np.argsort(s)
            s = s[order]
            w = w[order] / w_sum
            P = _compute_P_fast(s, w)
            if P > best_P:
                best_P = P
                best_s = s
                best_w = w

    return best_s, best_w, best_P


def _perturb_support(support, weights, n_new, s_max, deadline, rng):
    """Perturb the support: keep some points, add new random ones, then re-optimize weights."""
    n = len(support)
    s = support.copy()
    w = weights.copy()
    # pick a subset to keep (at least 2)
    keep = max(2, n - rng.integers(0, max(1, n - 2) + 1))
    idx = rng.choice(n, size=keep, replace=False)
    s_keep = s[idx]
    w_keep = w[idx]
    # add new random points
    n_add = max(0, n_new - len(s_keep))
    if n_add > 0:
        s_new = rng.uniform(0.0, s_max, size=n_add)
        w_new = rng.uniform(0.0, 1.0, size=n_add)
        s = np.concatenate([s_keep, s_new])
        w = np.concatenate([w_keep, w_new])
    order = np.argsort(s)
    s = s[order]
    w = w[order]
    w = np.maximum(w, 0.0)
    w_sum = w.sum()
    if w_sum <= 1e-9:
        w = np.full(len(s), 1.0 / len(s))
    else:
        w = w / w_sum
    opt_w, P = _optimize_weights_fixed_support(s, w, deadline)
    return s, opt_w, P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s kill

    best_P = -1.0
    best_support = None
    best_weights = None

    def _update(sup, wts, P):
        nonlocal best_P, best_support, best_weights
        if P > best_P:
            best_P = P
            best_support = np.asarray(sup, dtype=np.float64).ravel().copy()
            best_weights = np.asarray(wts, dtype=np.float64).ravel().copy()

    # ---- Phase 1: two-point analytic optimum ----
    p_opt, P_two = _optimize_two_point()
    _update(np.array([0.0, 1.0], dtype=np.float64),
            np.array([p_opt, 1.0 - p_opt], dtype=np.float64), P_two)

    if time.time() > deadline:
        return best_support, best_weights

    # ---- Phase 2: Differential Evolution joint search for several n ----
    s_max = 5.0
    for n in [3, 4, 5, 6, 8, 10, 12, 16]:
        if time.time() > deadline - 60:
            break
        # scale the two-point seed to this n: replicate the two values
        seed_s = np.zeros(n)
        seed_w = np.zeros(n)
        # put mass at 0 and 1, split evenly across the n slots
        seed_s[:n // 2] = 0.0
        seed_s[n // 2:] = 1.0
        seed_w[:n // 2] = p_opt / max(1, n // 2)
        seed_w[n // 2:] = (1.0 - p_opt) / max(1, n - n // 2)

        sup, wts, P = _de_search(n, s_max, deadline, seed_s, seed_w)
        if sup is not None and P > best_P:
            _update(sup, wts, P)

    if time.time() > deadline:
        return best_support, best_weights

    # ---- Phase 3: Refine weights of the current best support ----
    if best_support is not None and time.time() < deadline - 30:
        opt_w, P2 = _optimize_weights_fixed_support(best_support, best_weights, deadline)
        if P2 > best_P:
            _update(best_support, opt_w, P2)

    if time.time() > deadline:
        return best_support, best_weights

    # ---- Phase 4: Support perturbation + re-optimize (escape local optima) ----
    rng = np.random.default_rng(7)
    if best_support is not None:
        n_best = len(best_support)
        s_max = max(5.0, float(best_support.max()) * 2.0)
        n_perturb = 12
        for _ in range(n_perturb):
            if time.time() > deadline - 30:
                break
            n_new = int(rng.integers(max(3, n_best - 2), min(16, n_best + 4) + 1))
            sup, wts, P = _perturb_support(best_support, best_weights, n_new, s_max, deadline, rng)
            if P > best_P:
                _update(sup, wts, P)
        # final weight polish after perturbation
        if time.time() < deadline - 10:
            opt_w, P2 = _optimize_weights_fixed_support(best_support, best_weights, deadline)
            if P2 > best_P:
                _update(best_support, opt_w, P2)

    # ---- validate / fallback ----
    if best_support is None or best_weights is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        best_support = np.asarray(best_support, dtype=np.float64).ravel()
        best_weights = np.asarray(best_weights, dtype=np.float64).ravel()
        mask = (best_support >= -1e-9) & (best_weights >= -1e-9) & \
               np.isfinite(best_support) & np.isfinite(best_weights)
        best_support = best_support[mask]
        best_weights = best_weights[mask]
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        if len(best_support) == 0 or best_weights.sum() <= 1e-9:
            n = 32
            best_support = np.arange(n, dtype=np.float64)
            best_weights = np.ones(n, dtype=np.float64) / n

    return best_support, best_weights


# EVOLVE-BLOCK-END

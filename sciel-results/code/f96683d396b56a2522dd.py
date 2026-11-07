"""Convex epigraph formulation: minimize max cross-correlation of h and 1-h."""
import time
import numpy as np

N = 100
DEADLINE = 110.0  # safety margin before 120s kill; start clock at import

_t0 = time.time()


def _correlation_matrix():
    """Build matrix M so that corr_full = M @ h, where corr_full[k] =
    sum_i h[i] * (1 - h[i + k - N + 1]) for the 'full' correlation of h and (1-h).

    np.correlate(h, 1-h, mode='full')[k] = sum_i h[i] * (1-h)[i + k - (N-1)]
    Let j = i + k - (N-1). Then (1-h)[j] = 1 - h[j] (if in range).
    So term = h[i]*(1-h[j]) = h[i] - h[i]*h[j].
    The linear part in h: sum_i h[i] over valid i, minus quadratic h[i]h[j].
    """
    # We'll handle the objective directly via a function; matrix only for linear part.
    return None


def _objective_and_grad(x, t):
    """Objective: t (the max-correlation bound). We minimize t s.t. corr_k <= t for all k.
    Here we just compute the current max correlation for evaluation."""
    h = x
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))


def construct():
    elapsed = time.time() - _t0
    if elapsed > DEADLINE:
        return np.full(N, 0.5)

    # Start from a good initial guess: alternating 0/1 pattern (sum=50).
    # Alternating h = [1,0,1,0,...] gives sum=50 and strong anti-correlation.
    h0 = np.zeros(N)
    h0[0::2] = 1.0
    h0[1::2] = 0.0

    def _value(h):
        corr = np.correlate(h, 1.0 - h, mode="full")
        return float(np.max(corr)) / N * 2.0

    best_h = h0.copy()
    best_val = _value(best_h)

    # Epigraph convex optimization: minimize t subject to corr_k(h) <= t, 0<=h<=1, sum(h)=50.
    # corr_k(h) = sum_i h[i]*(1 - h[j]) where j = i + k - (N-1), only valid (in-range) terms.
    # This is h[i] - h[i]*h[j], which is NOT linear in h. But for fixed pattern structure it's convex-ish.
    # Use SLSQP on epigraph: variables [h (100), t (1)] = 101 vars.
    # Objective: t. Constraints: corr_k <= t for k=0..2N-2 (199 constraints), 0<=h<=1, sum(h)=50.

    from scipy.optimize import minimize

    def _corr_vector(h):
        return np.correlate(h, 1.0 - h, mode="full")

    n_corr = 2 * N - 1

    def _obj(z):
        return z[N]  # t

    def _obj_grad(z):
        g = np.zeros(N + 1)
        g[N] = 1.0
        return g

    def _constraint(z, k):
        h = z[:N]
        t = z[N]
        return t - _corr_vector(h)[k]

    cons = []
    for k in range(n_corr):
        cons.append({"type": "ineq", "fun": lambda z, k=k: _constraint(z, k)})
    cons.append({"type": "eq", "fun": lambda z: np.sum(z[:N]) - N / 2.0})

    bounds = [(0.0, 1.0)] * N + [(0.0, 10.0)]

    # Try SLSQP from the alternating start
    try:
        res = minimize(_obj, np.concatenate([h0, [best_val]]), jac=_obj_grad,
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-12})
        if res.success or res.fun < best_val:
            h_new = res.x[:N]
            h_new = np.clip(h_new, 0.0, 1.0)
            # Fix sum
            s = np.sum(h_new)
            if abs(s - N / 2.0) > 1e-9:
                h_new = h_new + (N / 2.0 - s) / N
                h_new = np.clip(h_new, 0.0, 1.0)
            v = _value(h_new)
            if v < best_val and np.isfinite(v):
                best_h = h_new
                best_val = v
    except Exception:
        pass

    elapsed = time.time() - _t0
    if elapsed > DEADLINE:
        return best_h

    # Try from a second start: block pattern (50 ones then 50 zeros)
    h1 = np.zeros(N)
    h1[:N // 2] = 1.0
    try:
        res2 = minimize(_obj, np.concatenate([h1, [_value(h1)]]), jac=_obj_grad,
                        method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 200, "ftol": 1e-12})
        if res2.fun < best_val:
            h_new = res2.x[:N]
            h_new = np.clip(h_new, 0.0, 1.0)
            s = np.sum(h_new)
            if abs(s - N / 2.0) > 1e-9:
                h_new = h_new + (N / 2.0 - s) / N
                h_new = np.clip(h_new, 0.0, 1.0)
            v = _value(h_new)
            if v < best_val and np.isfinite(v):
                best_h = h_new
                best_val = v
    except Exception:
        pass

    # Final validation
    h_final = np.clip(best_h, 0.0, 1.0)
    s = np.sum(h_final)
    if abs(s - N / 2.0) > 1e-9 * N:
        h_final = h_final + (N / 2.0 - s) / N
        h_final = np.clip(h_final, 0.0, 1.0)
    if not np.all(np.isfinite(h_final)):
        return np.full(N, 0.5)

    return h_final

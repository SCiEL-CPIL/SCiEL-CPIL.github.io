# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 620.0


def _P_of_support_weights(support, weights):
    """Exact P[X1+X2+X3 < 2*X4] for i.i.d. draws from the discrete measure.

    P = sum_{k,l} w_k w_l * P[X1+X2 < 2*s_l - s_k]
    where P[X1+X2 < t] = sum_{i,j} w_i w_j * 1[s_i + s_j < t]  (strict).
    """
    support = np.asarray(support, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = support.shape[0]
    if n == 0:
        return 0.0
    # all pairwise sums s_i + s_j (n x n)
    ss = support[:, None] + support[None, :]
    # threshold t_{k,l} = 2*s_l - s_k  (indexed [k, l])
    thr = 2.0 * support[None, :] - support[:, None]
    # strict inequality: count of pair-sums strictly below each threshold
    # sort pair-sums, use searchsorted 'left' for strict '<'
    ss_sorted = np.sort(ss, axis=None)
    counts = np.searchsorted(ss_sorted, thr, side='left')  # shape (n, n)
    # inner probability for each (k,l)
    inner = (weights[None, :] @ counts @ weights[:, None])  # (n, n)
    P = float((weights[None, :] * inner * weights[:, None]).sum())
    return P


def _valid_measure(support, weights):
    if support.size == 0 or weights.size == 0:
        return False
    if support.size != weights.size:
        return False
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        return False
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return False
    if float(weights.sum()) <= 1e-9:
        return False
    return True


def _compute_P(support, weights):
    if not _valid_measure(support, weights):
        return 0.0
    return _P_of_support_weights(support, weights)


def _logit(u):
    return np.log(u) - np.log1p(-u)


def _neg_P_logit(theta):
    n = int(theta.shape[0] // 2)
    if n <= 0:
        return 1e6
    s = theta[:n]
    logit_w = theta[n:]
    w = 1.0 / (1.0 + np.exp(-logit_w))
    P = _P_of_support_weights(s, w)
    return -P


def _optimize_from(support0, weights0, deadline):
    """Local optimization in (support, logit(weights)) space. Returns (support, weights, P)."""
    n = support0.shape[0]
    if n <= 0:
        return support0, weights0, 0.0
    s0 = np.maximum(support0, 0.0)
    w0 = np.clip(weights0, 1e-9, None)
    w0 = w0 / w0.sum()
    theta0 = np.concatenate([s0, _logit(w0)])
    best_theta = theta0.copy()
    best_P = _P_of_support_weights(s0, w0)
    try:
        res = minimize(_neg_P_logit, theta0, method='L-BFGS-B',
                       options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.x is not None:
            cand = res.x
            sc = cand[:n]
            wc = 1.0 / (1.0 + np.exp(-cand[n:]))
            if np.all(np.isfinite(sc)) and np.all(np.isfinite(wc)) and wc.sum() > 1e-9:
                sc = np.maximum(sc, 0.0)
                wc = wc / wc.sum()
                Pc = _P_of_support_weights(sc, wc)
                if Pc > best_P:
                    best_P = Pc
                    best_theta = cand
    except Exception:
        pass
    s_best = best_theta[:n]
    w_best = 1.0 / (1.0 + np.exp(-best_theta[n:]))
    w_best = w_best / w_best.sum()
    return s_best, w_best, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # ---- Baseline: two-point {0,1} with optimized p ----
    best_support = None
    best_weights = None
    best_P = -1.0

    # two-point analytic optimum
    p = (3.0 - np.sqrt(5.0)) / 2.0
    s2 = np.array([0.0, 1.0], dtype=np.float64)
    w2 = np.array([1.0 - p, p], dtype=np.float64)
    P2 = _compute_P(s2, w2)
    if P2 > best_P:
        best_P, best_support, best_weights = P2, s2.copy(), w2.copy()

    # ---- Local optimization from several starts, with time budget ----
    starts = []
    # start A: two-point
    starts.append((s2.copy(), w2.copy()))
    # start B: uniform on {0,1,2}
    starts.append((np.array([0.0, 1.0, 2.0]), np.array([1/3, 1/3, 1/3])))
    # start C: geometric-like 4 points
    r = 0.5
    sup = np.array([0.0, 1.0, 2.0, 4.0])
    wts = np.array([(1-r)*r**i for i in range(4)])
    starts.append((sup, wts / wts.sum()))
    # start D: geometric-like 6 points
    r = 0.4
    sup = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    wts = np.array([(1-r)*r**i for i in range(6)])
    starts.append((sup, wts / wts.sum()))
    # start E: uniform 0..4
    starts.append((np.arange(5, dtype=np.float64), np.ones(5)/5))

    for s0, w0 in starts:
        if time.time() > deadline - 30.0:
            break
        s, w, P = _optimize_from(s0, w0, deadline)
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()

    # ---- If we have spare time, try a couple more starts with slightly larger n ----
    if time.time() < deadline - 60.0:
        extra = []
        r = 0.35
        sup = np.array([(1.0 - r) / (1.0 - r**k) for k in range(1, 8)])
        wts = np.array([(1-r)*r**i for i in range(7)])
        extra.append((sup, wts / wts.sum()))
        # spread-out 0..6
        extra.append((np.arange(7, dtype=np.float64), np.ones(7)/7))
        for s0, w0 in extra:
            if time.time() > deadline - 30.0:
                break
            s, w, P = _optimize_from(s0, w0, deadline)
            if P > best_P:
                best_P = P
                best_support = s.copy()
                best_weights = w.copy()

    # ---- Final safety: ensure validity, else fall back to two-point ----
    if best_support is None or not _valid_measure(best_support, best_weights):
        best_support = s2.copy()
        best_weights = w2.copy()
    else:
        # clip tiny negatives from optimizer
        best_support = np.maximum(best_support, 0.0)
        best_weights = np.maximum(best_weights, 0.0)
        if best_weights.sum() <= 1e-9:
            best_support = s2.copy()
            best_weights = w2.copy()
        else:
            best_weights = best_weights / best_weights.sum()

    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END

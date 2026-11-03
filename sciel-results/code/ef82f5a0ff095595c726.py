import numpy as np
from scipy.optimize import minimize
import time
import warnings

warnings.filterwarnings('ignore')

N = 50
DW = 1.0 / N

# ---------------------------------------------------------------------------
# Exact continuous autoconvolution kernel for step functions on [-1/4, 1/4]
# ---------------------------------------------------------------------------
def _precompute_conv_kernel():
    x = np.linspace(-0.25, 0.25, N + 1)
    n_pts = 4 * N  # quadrature points on [-0.5, 0.5]
    t_grid = np.linspace(-0.5, 0.5, n_pts + 1)
    midpoints = (t_grid[:-1] + t_grid[1:]) / 2.0

    i_idx = np.arange(N)[:, None, None]
    j_idx = np.arange(N)[None, :, None]
    k_idx = np.arange(n_pts)[None, None, :]

    x_i = x[i_idx]
    x_i1 = x[i_idx + 1]
    x_j = x[j_idx]
    x_j1 = x[j_idx + 1]
    t_k = midpoints[k_idx]

    lo = np.maximum(x_i, t_k - x_j1)
    hi = np.minimum(x_i1, t_k - x_j)
    t_lo = x_i + x_j
    t_hi = x_i1 + x_j1
    valid = (t_k >= t_lo) & (t_k <= t_hi)
    K = np.where(valid, np.maximum(0.0, hi - lo), 0.0)
    return K.reshape(N * N, -1), midpoints

_K_FLAT, _T_MID = _precompute_conv_kernel()


def _autoconv_values(h):
    """Exact autoconvolution of step function at quadrature points."""
    outer = np.outer(h, h).ravel()
    return outer @ _K_FLAT


def _compute_score(h):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    L1 = np.sum(h) * DW
    if L1 < 1e-15:
        return 0.0
    L2_sq = np.sum(h * h) * DW
    ac = _autoconv_values(h)
    sup = np.max(np.abs(ac))
    if sup < 1e-15:
        return 0.0
    return L2_sq / (L1 * sup)


def _neg_smooth(h, beta):
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    L1 = np.sum(h) * DW
    if L1 < 1e-15:
        return 0.0
    L2_sq = np.sum(h * h) * DW
    ac = _autoconv_values(h)
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-15:
        return 0.0
    shifted = beta * (a - m)
    logsum = m + np.log(np.sum(np.exp(shifted))) / beta
    if logsum < 1e-15:
        return 0.0
    return -(L2_sq / (L1 * logsum))


def _neg_exact(h):
    return -_compute_score(h)


def _try_update(h, best):
    h = np.maximum(h, 0.0)
    if np.sum(h) < 1e-15:
        return best
    sc = _compute_score(h)
    if sc > best[1] + 1e-12:
        return [h.copy(), sc]
    return best


def _lbgfb(start, beta, maxiter, deadline):
    try:
        res = minimize(_neg_smooth, start, method='L-BFGS-B',
                       bounds=[(0.0, 20.0)] * N, args=(beta,),
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-9})
        return np.maximum(res.x, 0.0)
    except Exception:
        return start


def _exact_lbgfb(start, maxiter, deadline):
    try:
        res = minimize(_neg_exact, start, method='L-BFGS-B',
                       bounds=[(0.0, 20.0)] * N,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-9})
        return np.maximum(res.x, 0.0)
    except Exception:
        return start


def _coordinate_refine(h, deadline, passes=30):
    h = h.copy()
    cur = _compute_score(h)
    for _ in range(passes):
        if time.time() > deadline:
            break
        improved = False
        for j in range(N):
            if time.time() > deadline:
                break
            others = h.copy()
            others[j] = 0.0
            base_sum = np.sum(others)
            if base_sum < 1e-15:
                continue
            # candidate heights: current, zero, and a geometric spread
            scale = max(1.0, base_sum)
            cands = np.unique(np.concatenate([
                [h[j], 0.0],
                np.geomspace(1e-4 * scale, 8.0 * scale, 24),
            ]))
            best_hj = h[j]
            best_sc = cur
            for hj in cands:
                h_test = others.copy()
                h_test[j] = hj
                sc = _compute_score(h_test)
                if sc > best_sc + 1e-12:
                    best_sc = sc
                    best_hj = hj
            if abs(best_hj - h[j]) > 1e-9:
                h[j] = best_hj
                cur = best_sc
                improved = True
        if not improved:
            break
    return h, cur


def _support_search(rng, deadline):
    """Explore which bins are active (support) as a combinatorial variable."""
    best = [None, -1.0]
    # Candidate supports: contiguous blocks, pairs, random sparse sets
    supports = []
    # contiguous blocks of various lengths
    for length in [2, 3, 4, 5, 6, 8, 10]:
        for start in range(0, N - length + 1, max(1, (N - length) // 6)):
            supports.append(list(range(start, start + length)))
    # a few random sparse supports
    for _ in range(20):
        k = int(rng.integers(2, 9))
        supports.append(sorted(rng.choice(N, size=k, replace=False).tolist()))

    for sup in supports:
        if time.time() > deadline:
            break
        # initialize heights on support
        base = np.zeros(N)
        for idx in sup:
            base[idx] = 1.0
        base = base / np.sum(base) * len(sup)
        # smooth optimize restricted to support (others forced 0 via bounds)
        bounds = [(0.0, 20.0) if i in sup else (0.0, 0.0) for i in range(N)]
        try:
            res = minimize(_neg_smooth, base, method='L-BFGS-B', bounds=bounds,
                           args=(80.0,),
                           options={'maxiter': 120, 'ftol': 1e-12, 'gtol': 1e-8})
            h = np.maximum(res.x, 0.0)
            best = _try_update(h, best)
        except Exception:
            pass
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    rng = np.random.default_rng(7)

    def time_ok():
        return time.time() < deadline

    # Initial diverse starts
    starts = []
    h2 = np.zeros(N); h2[0] = 2.0 / 3.0; h2[1] = 1.0 / 3.0
    starts.append(h2)
    h3 = np.zeros(N); h3[0] = 0.5; h3[1] = 0.3; h3[2] = 0.2
    starts.append(h3)
    h5 = np.zeros(N); h5[:5] = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    starts.append(h5)
    starts.append(np.ones(N) / N)
    # a couple random starts
    for _ in range(3):
        hr = rng.random(N)
        hr = hr / np.sum(hr) * N
        starts.append(hr)

    best = [None, -1.0]
    for s in starts:
        best = _try_update(s, best)

    # Phase 1: support-selection search (structural: choose active bins)
    if time_ok():
        sb = _support_search(rng, deadline - 30.0)
        if sb[1] > best[1]:
            best = sb

    # Phase 2: smooth surrogate L-BFGS-B from current best and a few starts
    cur = best[0].copy() if best[0] is not None else h2.copy()
    for beta in [50.0, 150.0, 400.0]:
        if not time_ok():
            break
        trial = _lbgfb(cur, beta, 250, deadline)
        best = _try_update(trial, best)
        cur = best[0]
    # also run smooth from a couple other starts
    for s in starts[:3]:
        if not time_ok():
            break
        trial = _lbgfb(s, 150.0, 150, deadline)
        best = _try_update(trial, best)
        cur = best[0]

    # Phase 3: exact-objective L-BFGS-B
    if time_ok():
        trial = _exact_lbgfb(cur, 400, deadline)
        best = _try_update(trial, best)
        cur = best[0]

    # Phase 4: coordinate refinement (fast, exact)
    if time_ok():
        h_ref, sc_ref = _coordinate_refine(cur, deadline, passes=40)
        best = _try_update(h_ref, best)

    if best[0] is None:
        best[0] = h2.copy()

    return [float(x) for x in best[0]]

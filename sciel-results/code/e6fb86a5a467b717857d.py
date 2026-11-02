import numpy as np
import time
import warnings

N = 50

# EVOLVE-BLOCK-START
def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _score(h):
    h = np.asarray(h, dtype=float)
    s = h.sum()
    if s <= 0 or np.any(h < 0):
        return 0.0
    h = h / s
    l2sq = float(np.sum(h * h))
    conv = _autoconv(h)
    m = float(np.max(conv))
    if m <= 0:
        return 0.0
    return l2sq / m

def _sidon_supports(max_m=12, seed=0):
    """Generate several structured supports of varying size (Sidon-like)."""
    rng = np.random.RandomState(seed)
    supports = []
    # Small explicit supports
    for m in range(2, max_m + 1):
        # random sorted support
        sup = np.sort(rng.choice(N, size=m, replace=False))
        supports.append(sup)
    # Arithmetic-like supports (consecutive)
    for start in range(0, N - 6, 3):
        supports.append(np.arange(start, start + 8))
    # Geometric-ish
    g = np.array([0, 1, 3, 7, 15, 31])
    supports.append(g[g < N])
    # Bose-Chowla style: powers of 2 mod structure (just use sparse)
    supports.append(np.array([0, 1, 4, 10, 21, 43] if 43 < N else [0, 1, 4, 10, 21]))
    return supports

def _reweight_on_support(h, support, peak_idx, n_iter=200, tol=1e-12):
    """Multiplicative reweighting: fix peak at peak_idx, push mass to reduce
    other lags relative to the chosen peak while keeping h nonnegative."""
    m = len(support)
    idx = np.zeros(N, dtype=int)
    idx[support] = np.arange(m)
    # work in local coordinates
    x = h[support].copy()
    x = np.maximum(x, 1e-12)
    x = x / x.sum()
    # The chosen peak lag in full-conv coordinates
    # full conv index = i + j + (m-1)  ... but we use global N-indexed conv
    # We'll just compute global conv and find its argmax each iter, but bias toward peak_idx
    for it in range(n_iter):
        full = np.zeros(N)
        full[support] = x
        conv = _autoconv(full)
        # target: make conv[peak_idx] the max
        cur_peak = conv[peak_idx]
        if cur_peak <= 0:
            break
        # ratio of each lag to the target peak
        ratio = conv / cur_peak
        # damp lags that exceed the target (shouldn't if peak_idx is max)
        # multiplicative update: increase x where it contributes to peak_idx,
        # decrease where it contributes to over-peak lags
        grad = np.zeros(N)
        # contribution of x_j to conv[k] = sum_j x_j x_{j+k}
        # We approximate by: for the target lag, we want to boost the pair
        # Use simple: x <- x * (1 + eps * (target_contribution - other_contribution))
        # target contribution of x_j = sum over k: x_{k} where j+k=peak or j=peak+k
        tc = np.zeros(N)
        oc = np.zeros(N)
        for j in range(N):
            # j contributes to conv[peak_idx] via partner peak_idx - j or j - peak_idx
            p1 = peak_idx - j
            p2 = j - peak_idx
            if 0 <= p1 < N:
                tc[j] += full[p1]
            if 0 <= p2 < N:
                tc[j] += full[p2]
            # other lags: sum over all k != peak_idx
            for k in range(len(conv)):
                if k == peak_idx:
                    continue
                p1 = k - j
                p2 = j - k
                c = 0.0
                if 0 <= p1 < N:
                    c += full[p1]
                if 0 <= p2 < N:
                    c += full[p2]
                oc[j] += c * (conv[k] / cur_peak)
        diff = tc - oc
        x_new = x * (1.0 + 0.05 * diff[support])
        x_new = np.maximum(x_new, 1e-14)
        x_new = x_new / x_new.sum()
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    out = np.zeros(N)
    out[support] = x
    return out

def _coordinate_polish(h, n_passes=300, tol=1e-12):
    """1-D coordinate ascent on each height (keeping others fixed), respecting
    nonnegativity, to locally maximize the score."""
    h = np.asarray(h, dtype=float).copy()
    s = h.sum()
    if s <= 0:
        return h
    best = _score(h)
    for _ in range(n_passes):
        improved = False
        for i in range(N):
            if h[i] < 1e-15 and best > 1.15:
                continue  # keep sparse
            # try a few candidate values for h[i]
            others = h.copy()
            others[i] = 0.0
            base_sum = others.sum()
            cands = [h[i]]
            if h[i] > 0:
                cands += [h[i] * 0.5, h[i] * 1.5, h[i] * 0.1, h[i] * 3.0, 0.0]
            else:
                cands += [base_sum * 0.01, base_sum * 0.05, base_sum * 0.2]
            cands = [max(c, 0.0) for c in cands]
            cur_best = best
            cur_val = h[i]
            for c in cands:
                test = others.copy()
                test[i] = c
                sc = _score(test)
                if sc > cur_best + 1e-12:
                    cur_best = sc
                    cur_val = c
                    improved = True
            h[i] = cur_val
        if not improved:
            break
    return h

def _diverse_starts():
    starts = []
    rng = np.random.RandomState(123)
    # concentrated pairs
    for a in [0.8, 0.9, 0.95, 0.7, 0.6]:
        v = np.zeros(N)
        v[0] = a
        v[1] = 1 - a
        starts.append(v)
    # triplets
    for a, b in [(0.6, 0.3), (0.7, 0.2), (0.5, 0.4), (0.8, 0.1), (0.4, 0.4)]:
        v = np.zeros(N)
        v[0] = a
        v[1] = b
        v[2] = 1 - a - b
        if v[2] >= 0:
            starts.append(v)
    # random sparse
    for _ in range(20):
        m = rng.randint(2, 8)
        sup = np.sort(rng.choice(N, size=m, replace=False))
        v = np.zeros(N)
        w = rng.rand(m)
        v[sup] = w / w.sum()
        starts.append(v)
    # random dense
    for _ in range(5):
        v = rng.rand(N)
        v = v / v.sum()
        starts.append(v)
    # geometric
    for r in [0.5, 0.6, 0.7, 0.8]:
        v = np.zeros(N)
        val = 1.0
        for i in range(10):
            v[i] = val
            val *= r
        v = v / v.sum()
        starts.append(v)
    return starts

def construct():
    DEADLINE = time.time() + 135
    best_h = None
    best_score = -1.0

    def consider(h):
        nonlocal best_h, best_score
        h = np.maximum(np.asarray(h, dtype=float), 0.0)
        s = h.sum()
        if s <= 0:
            return
        h = h / s
        sc = _score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # 1) evaluate diverse starts
    for v in _diverse_starts():
        consider(v)
    if best_h is None:
        best_h = np.zeros(N); best_h[0] = 1.0

    # 2) reweighting on structured supports with various peak indices
    supports = _sidon_supports(max_m=10, seed=1)
    for sup in supports:
        if time.time() > DEADLINE - 30:
            break
        m = len(sup)
        # candidate peak indices: lags corresponding to differences within support
        diffs = set()
        for a in sup:
            for b in sup:
                if a != b:
                    diffs.add(abs(int(a) - int(b)))
        # full-conv index for lag d is d (since full conv index k corresponds to
        # shift k - (m-1)... but with global N support, conv index = i+j)
        # We'll just try a spread of peak indices
        peak_candidates = sorted(diffs) + [0, 1, 2]
        for pk in peak_candidates:
            if time.time() > DEADLINE - 25:
                break
            # start: uniform on support
            v = np.zeros(N)
            v[sup] = 1.0 / m
            # reweight to favor peak pk
            v2 = _reweight_on_support(v, sup, pk, n_iter=150)
            consider(v2)
            # coordinate polish the result
            if time.time() < DEADLINE - 15:
                v3 = _coordinate_polish(v2, n_passes=100)
                consider(v3)

    # 3) coordinate polish the global best
    if time.time() < DEADLINE - 10:
        polished = _coordinate_polish(best_h, n_passes=400)
        consider(polished)

    # 4) second polish round
    if time.time() < DEADLINE - 5:
        polished2 = _coordinate_polish(best_h, n_passes=300)
        consider(polished2)

    best_h = np.maximum(best_h, 0.0)
    s = best_h.sum()
    if s <= 0:
        best_h = np.zeros(N); best_h[0] = 1.0
    else:
        best_h = best_h / s
    return best_h
# EVOLVE-BLOCK-END

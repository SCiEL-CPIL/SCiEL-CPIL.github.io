import numpy as np
import time

N = 50
DELTA = 1.0 / 100.0  # interval width if needed

def _autoconv(h):
    return np.convolve(h, h, mode='full')

def _proxy_score(h):
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    h = h / s
    c = _autoconv(h)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    max_c = np.max(c)
    if max_c < 1e-12:
        return 0.0
    return l2_sq / (l1 * max_c)

def _smooth_score(h, tau=0.01):
    # Use log-sum-exp to smooth the max
    s = np.sum(h)
    if s < 1e-12:
        return 0.0
    h = h / s
    c = _autoconv(h)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    # soft max
    cmax = np.max(c)
    c_shifted = c - cmax
    soft_max = cmax + tau * np.log(np.sum(np.exp(c_shifted / tau)))
    if soft_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * soft_max)

def _gradient_smooth(h, tau=0.01):
    # Gradient of smooth score w.r.t. h (before normalization)
    # We'll do it numerically for safety, but that's slow. Use analytic.
    s = np.sum(h)
    h_n = h / s
    c = _autoconv(h_n)
    l2_sq = np.sum(h_n**2)
    l1 = np.sum(h_n)
    cmax = np.max(c)
    c_shifted = c - cmax
    soft_max = cmax + tau * np.log(np.sum(np.exp(c_shifted / tau)))
    # d(soft_max)/dc
    weights = np.exp(c_shifted / tau) / np.sum(np.exp(c_shifted / tau))
    # dc/dh_n: c[k] = sum_i h_n[i] h_n[k-i]
    # This is the gradient of the quadratic form.
    # d(soft_max)/dh_n = sum_k weights[k] * dc[k]/dh_n
    # dc[k]/dh_n[j] = h_n[j + k] + h_n[k - j] (with bounds)
    # We can compute via correlation.
    # Use the fact that grad of c w.r.t. h_n is [h_n, h_n] cross-correlation.
    # Let's compute numerically-free: 
    # dc/dh_n[j] = sum_k weights[k] * (h_n[j+k] + h_n[k-j]) where indices valid.
    # This equals weights convolved with h_n plus h_n convolved with weights.
    g = np.zeros(N)
    # c has length 2N-1. weights same.
    # For each j, dc/dh_n[j] = sum_k w[k] * (h_n[j+k] if j+k<N else 0 + h_n[k-j] if k-j>=0 else 0)
    # = (w * h_n reversed?) Let's just do it with a loop over k for clarity (N=50, 99 terms, fine).
    for k in range(len(c)):
        wk = weights[k]
        for j in range(N):
            # contribution from h_n[j] to c[k]: c[k] includes h_n[j]*h_n[k-j] and h_n[k-j]*h_n[j]
            # Actually c[k] = sum_i h_n[i] h_n[k-i]. Term with i=j: h_n[j] h_n[k-j] (if k-j in [0,N-1])
            # So dc[k]/dh_n[j] = h_n[k-j] (if 0<=k-j<N)
            if 0 <= k - j < N:
                g[j] += wk * h_n[k - j]
    # Now score = l2_sq / (l1 * soft_max). With l1=1 (normalized), score = l2_sq / soft_max.
    # d(score)/dh_n = (d(l2_sq)/dh_n * soft_max - l2_sq * d(soft_max)/dh_n) / soft_max^2
    # d(l2_sq)/dh_n = 2 h_n
    d_score_dn = (2 * h_n * soft_max - l2_sq * g) / (soft_max ** 2)
    # Chain rule: h_n = h/s, dh_n/dh = (1/s) I - (1/s^2) 1 1^T
    # For simplicity, since we project/normalize each step, use d_score_dn as gradient direction in h_n space,
    # then map back: grad_h = d_score_dn / s - (sum(d_score_dn * h_n)/s^2) * 1 ... 
    # Simpler: just return d_score_dn and handle normalization in the update by re-normalizing.
    return d_score_dn

def _projected_gradient_step(h, grad, step):
    h_new = h + step * grad
    h_new = np.maximum(h_new, 0)
    s = np.sum(h_new)
    if s < 1e-12:
        return h
    return h_new / s

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def time_ok():
        return time.time() < deadline

    best_h = None
    best_score = -1.0

    def try_update(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-12:
            return
        h = h / s
        sc = _proxy_score(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # Initial starts
    starts = []
    # [2/3, 1/3, 0...]
    h0 = np.zeros(N); h0[0] = 2.0/3.0; h0[1] = 1.0/3.0
    starts.append(h0)
    # uniform
    h1 = np.ones(N) / N
    starts.append(h1)
    # random sparse
    rng = np.random.default_rng(42)
    for _ in range(5):
        hr = np.zeros(N)
        k = rng.integers(2, 15)
        idx = rng.choice(N, size=k, replace=False)
        hr[idx] = rng.random(k)
        starts.append(hr)
    # random dense
    for _ in range(5):
        hr = rng.random(N)
        starts.append(hr)

    tau = 0.02
    for h_start in starts:
        if not time_ok():
            break
        h = h_start.copy()
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-12:
            continue
        h = h / s
        try_update(h)

        # Adaptive step size projected gradient ascent
        step = 0.1
        prev_score = _proxy_score(h)
        for it in range(2000):
            if not time_ok():
                break
            grad = _gradient_smooth(h, tau)
            h_new = _projected_gradient_step(h, grad, step)
            sc_new = _proxy_score(h_new)
            if sc_new > prev_score:
                h = h_new
                prev_score = sc_new
                step = min(step * 1.05, 1.0)
                try_update(h)
            else:
                step = max(step * 0.5, 1e-4)
                if step < 1e-5:
                    break

        # Coordinate ascent refinement
        if time_ok():
            h = h.copy()
            for coord_pass in range(50):
                if not time_ok():
                    break
                improved = False
                for j in range(N):
                    if not time_ok():
                        break
                    # Optimize h[j] with others fixed
                    others = h.copy()
                    others[j] = 0.0
                    base_sum = np.sum(others)
                    base_l2 = np.sum(others**2)
                    # c depends on h[j]
                    # We'll do a 1D search over h[j] in [0, 10*mean]
                    # score(hj) = (base_l2 + hj^2) / ((base_sum + hj) * max_c(hj))
                    # max_c is piecewise; just sample.
                    best_hj = h[j]
                    best_sc = _proxy_score(h)
                    # sample grid
                    scale = max(1.0, base_sum)
                    grid = np.linspace(0, 5*scale, 40)
                    for hj in grid:
                        h_test = others.copy()
                        h_test[j] = hj
                        sc = _proxy_score(h_test)
                        if sc > best_sc:
                            best_sc = sc
                            best_hj = hj
                    if best_hj != h[j]:
                        h[j] = best_hj
                        improved = True
                if not improved:
                    break
            try_update(h)

    if best_h is None:
        best_h = np.zeros(N); best_h[0] = 2.0/3.0; best_h[1] = 1.0/3.0

    return [float(x) for x in best_h]

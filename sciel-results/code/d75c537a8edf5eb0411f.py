import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    """Compute J(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    if np.max(h) < 1e-15:
        return 0.0
    # Scale invariant, but use raw for computation
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Autoconvolution via FFT for speed
    n_fft = 2 * N  # full conv needs 2N-1, round up to power of 2 for speed
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2*N-1]
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _neg_ratio(h):
    return -_compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    best_h = None
    best_ratio = 0.0
    
    def update_best(h, r):
        nonlocal best_h, best_ratio
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Candidate 1: Single spike
    h1 = np.zeros(N)
    h1[0] = 1.0
    r1 = _compute_ratio(h1)
    update_best(h1, r1)
    
    # Candidate 2: Two spikes at max separation
    for d in [1, 2, 3, 5, 7, 10, 13, 17, 25]:
        if d >= N:
            continue
        h2 = np.zeros(N)
        h2[0] = 1.0
        h2[d] = 1.0
        r2 = _compute_ratio(h2)
        update_best(h2, r2)
    
    # Candidate 3: Three spikes
    for i in range(0, N, 2):
        for j in range(i+1, N, 2):
            for k in range(j+1, N, 2):
                if time.time() > deadline:
                    break
                h3 = np.zeros(N)
                h3[i] = 1.0
                h3[j] = 1.0
                h3[k] = 1.0
                r3 = _compute_ratio(h3)
                update_best(h3, r3)
        if time.time() > deadline:
            break
    
    # Random sparse binary search
    rng = np.random.RandomState(42)
    for trial in range(3000):
        if time.time() > deadline:
            break
        k = rng.randint(1, 12)
        indices = rng.choice(N, k, replace=False)
        hr = np.zeros(N)
        hr[indices] = 1.0
        rr = _compute_ratio(hr)
        update_best(hr, rr)
    
    # Local search / polish on best candidate
    if best_h is not None:
        h_cur = best_h.copy()
        # Nelder-Mead optimization
        from scipy.optimize import minimize
        res = minimize(_neg_ratio, h_cur, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        if time.time() < deadline:
            h_opt = np.clip(res.x, 0, None)
            r_opt = _compute_ratio(h_opt)
            update_best(h_opt, r_opt)
        
        # Additional local perturbation search
        h_cur = best_h.copy()
        for it in range(10000):
            if time.time() > deadline:
                break
            i = rng.randint(N)
            old = h_cur[i]
            h_cur[i] = max(0.0, h_cur[i] + rng.uniform(-0.1, 0.1))
            r = _compute_ratio(h_cur)
            if r > best_ratio:
                best_ratio = r
                best_h = h_cur.copy()
            else:
                h_cur[i] = old
    
    # Final: try optimizing a few promising binary patterns with continuous weights
    if best_h is not None and time.time() < deadline - 10:
        # Take the support of best and optimize weights
        support = np.where(best_h > 1e-6)[0]
        if len(support) >= 2:
            h_w = np.zeros(N)
            h_w[support] = 1.0
            res2 = minimize(_neg_ratio, h_w, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11,
                                    'adaptive': True})
            h_opt2 = np.clip(res2.x, 0, None)
            r_opt2 = _compute_ratio(h_opt2)
            update_best(h_opt2, r_opt2)
    
    # Ensure nonnegative
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    best_h = np.clip(best_h, 0, None)
    return list(best_h)

import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)  # 99
    y = np.concatenate(([0.0], g, [0.0]))  # length m+2 = 101
    # l2 = sum over i in 0..m of (w/3)*(y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # i ranges 0..m, i.e., 0..99 (since m=99), y[i] and y[i+1] valid for y of length 101
    sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value(log_h):
    h = np.exp(log_h)
    return -compute_value(h)

def neg_value_direct(h):
    return -compute_value(h)

def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # should be 1.0

    rng = np.random.RandomState(42)

    def try_optimize_log(start_log_h, maxiter):
        nonlocal best_h, best_val
        t0 = time.time()
        res = minimize(
            neg_value,
            start_log_h,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        h_cand = np.exp(res.x)
        v_cand = compute_value(h_cand)
        if v_cand > best_val:
            best_val = v_cand
            best_h = h_cand.copy()
        return time.time() - t0

    def try_optimize_direct(start_h, maxiter):
        nonlocal best_h, best_val
        t0 = time.time()
        res = minimize(
            neg_value_direct,
            start_h,
            method='L-BFGS-B',
            bounds=[(1e-6, None)] * N,
            options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        h_cand = np.maximum(res.x, 0.0)
        v_cand = compute_value(h_cand)
        if v_cand > best_val:
            best_val = v_cand
            best_h = h_cand.copy()
        return time.time() - t0

    # Start from uniform (log 0 = 0)
    try_optimize_log(np.zeros(N), maxiter=3000)

    # Try a few multi-start perturbations if time remains
    n_starts = 12
    for i in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        # Perturb: smooth random log-heights, centered so mean ~0
        x = rng.randn(N)
        # light smoothing
        kernel = np.ones(5)/5.0
        x = np.convolve(x, kernel, mode='same')
        x = x - x.mean()
        x = 0.5 * x + 0.5 * np.zeros(N)  # damp toward uniform
        # Add a few structured shapes
        if i == 1:
            x = np.sin(np.pi * np.arange(N) / N) + 0.3
        elif i == 2:
            x = np.cos(2*np.pi*np.arange(N)/N) * 0.5 + 0.5
        elif i == 3:
            x = 1 - 2*np.abs(np.arange(N)/N - 0.5)
        elif i == 4:
            x = np.exp(-((np.arange(N)-N//2)**2)/30.0)
            x = x - x.mean() + 0.5
        elif i == 5:
            x = np.where(np.arange(N) < N//2, 0.2, 0.8) + 0.1*rng.randn(N)
        elif i == 6:
            # Triangle peak
            x = 1 - 2*np.abs(np.arange(N)/N - 0.5)
            x = x**2
        elif i == 7:
            # Two peaks
            x = np.exp(-((np.arange(N)-N//4)**2)/10.0) + np.exp(-((np.arange(N)-3*N//4)**2)/10.0)
            x = x - x.mean() + 0.5
        elif i == 8:
            # Ramp
            x = np.linspace(0.2, 0.8, N)
        elif i == 9:
            # Alternating
            x = 0.5 + 0.3*np.cos(4*np.pi*np.arange(N)/N)
        else:
            x = x + 0.2*rng.randn(N)
        x = np.maximum(x, -5.0)  # avoid underflow
        try_optimize_log(x, maxiter=2000)

    # Final polish: direct optimization from best found
    if time.time() < deadline - 20.0:
        try_optimize_direct(best_h.copy(), maxiter=5000)

    # One more log-space polish
    if time.time() < deadline - 10.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-6)), maxiter=3000)

    return list(np.asarray(best_h, dtype=float))

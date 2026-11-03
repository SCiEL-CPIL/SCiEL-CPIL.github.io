"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combined approach:
 - FFT-based overlap computation (fast)
 - Smooth-max (log-sum-exp) surrogate with analytic gradient
 - Structured initial candidates (cosine/sine/chirp/square/triangle)
 - Optimization via SLSQP (handles box+sum constraints cleanly) with
   multiple beta values and multiple starts
 - PGD with exact projection onto {0<=x<=1, sum=N/2} for refinement
 - Deadline guard to finish well before 150s
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 142  # safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_val_grad(h, beta):
    """
    Value and gradient of the smooth maximum (log-sum-exp) of C_k.
    Returns (value, gradient).
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)

    # Numerically stable log-sum-exp
    shifted = beta * (cs - m)
    shifted = np.clip(shifted, -500, 500)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)

    if sum_exp == 0:
        w = np.full(n, 1.0 / n)
        val = m
    else:
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta

    grad = np.zeros(n)

    # C_k = 1 - (2/n) * r_k where r_k = sum_{i=k}^{n-1} h_i h_{i-k}
    # dC_k/dh_j = -(2/n) * (h_{j-k} [if j>=k] + h_{j+k} [if j+k<n])

    for k in range(n):
        if w[k] < 1e-14:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n - k]
        grad[:n - k] += factor * h[k:]

    return val, grad


def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    using bisection on a shift parameter.
    """
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0

    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum

    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid

    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)


def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0, 1]."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)

    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)

    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)

    return h


def _run_slsqp(h0, beta, maxiter, ftol, deadline):
    """Run SLSQP optimization with smooth max objective."""
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
             'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N

    def obj_func(x):
        val, _ = _smooth_max_val_grad(x, beta)
        return val

    def jac_func(x):
        _, grad = _smooth_max_val_grad(x, beta)
        return grad

    try:
        res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.x is not None:
            return res.x
        return h0
    except Exception:
        return h0


def _pgd_optimize(h0, beta, steps, lr, deadline):
    """
    Projected Gradient Descent optimization with backtracking line search.
    Uses exact projection onto {0<=x<=1, sum=N/2}.
    """
    h = h0.copy()
    target_sum = N / 2.0
    h = _project_sum_box(h, target_sum)
    current_lr = lr
    prev_val, _ = _smooth_max_grad(h, beta)

    for i in range(steps):
        if time.time() > deadline:
            break

        val, grad = _smooth_max_grad(h, beta)

        step_lr = current_lr
        new_h = h - step_lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _smooth_max_grad(new_h_proj, beta)

        # Backtracking
        while new_val > prev_val and step_lr > 1e-10:
            step_lr /= 2.0
            new_h = h - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)

        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            if i % 10 == 0:
                current_lr = min(current_lr * 1.05, 0.1)
        else:
            current_lr = max(current_lr * 0.5, 1e-8)

    return h


def _try_analytical_form():
    """
    Try structured forms that might be near-optimal.
    """
    candidates = []
    i = np.arange(N)

    # Cosine with fundamental frequency
    for amp in [0.3, 0.4, 0.45, 0.5, 0.55]:
        h = 0.5 + amp * np.cos(2.0 * np.pi * i / N)
        candidates.append(h)

    # Sine with fundamental frequency
    for amp in [0.3, 0.4, 0.45, 0.5]:
        h = 0.5 + amp * np.sin(2.0 * np.pi * i / N)
        candidates.append(h)

    # Second harmonic
    for amp in [0.2, 0.3, 0.4]:
        h = 0.5 + amp * np.cos(4.0 * np.pi * i / N)
        candidates.append(h)

    # Mix of harmonics
    h = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / N) + 0.15 * np.cos(4.0 * np.pi * i / N)
    candidates.append(h)

    h = 0.5 + 0.25 * np.cos(2.0 * np.pi * i / N) - 0.1 * np.cos(4.0 * np.pi * i / N)
    candidates.append(h)

    # Square-wave-like
    h = np.where(i % 2 == 0, 0.7, 0.3)
    candidates.append(h)

    h = np.where(i % 2 == 0, 0.6, 0.4)
    candidates.append(h)

    # Triangular wave
    phase = 2.0 * np.pi * i / N
    h = 0.5 + (2.0 / np.pi) * np.arcsin(np.sin(phase)) * 0.4
    candidates.append(h)

    # Chirps (quadratic phase)
    for alpha in [0.5, 1.0, 2.0]:
        for phase in [0, np.pi / 2]:
            h_chirp = 0.5 + 0.4 * np.cos(2.0 * np.pi * alpha * (i ** 2) / N + phase)
            candidates.append(h_chirp)

    # Higher frequency cosines
    for freq in [3, 5, 10]:
        h = 0.5 + 0.3 * np.cos(2.0 * np.pi * freq * i / N)
        candidates.append(h)

    return candidates


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        if h is None:
            return
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # 1. Generate initial candidates
    candidates = []

    # Constant
    candidates.append(np.full(N, 0.5))

    # Analytical structured forms
    candidates.extend(_try_analytical_form())

    # Random starts
    for _ in range(10):
        h = np.random.uniform(0.05, 0.95, N)
        candidates.append(h)

    # Evaluate all candidates
    for h0 in candidates:
        _update_best(h0)

    if best_h is None:
        best_h = np.full(N, 0.5)

    # 2. Multi-phase optimization
    # Phase 1: SLSQP with low beta for broad exploration
    if time.time() - t0 < _DEADLINE - 60:
        for beta in [10.0, 20.0, 40.0]:
            if time.time() - t0 > _DEADLINE - 55:
                break
            h_opt = _run_slsqp(best_h, beta=beta, maxiter=150, ftol=1e-7,
                               deadline=_DEADLINE - 50)
            _update_best(h_opt)

    # Phase 2: PGD with medium beta
    if time.time() - t0 < _DEADLINE - 35:
        h_opt = _pgd_optimize(best_h, beta=50.0, steps=500, lr=0.02,
                              deadline=_DEADLINE - 35)
        _update_best(h_opt)

    # Phase 3: SLSQP with higher beta for refinement
    if time.time() - t0 < _DEADLINE - 20:
        for beta in [100.0, 200.0]:
            if time.time() - t0 > _DEADLINE - 18:
                break
            h_opt = _run_slsqp(best_h, beta=beta, maxiter=200, ftol=1e-9,
                               deadline=_DEADLINE - 15)
            _update_best(h_opt)

    # Phase 4: PGD with high beta for polishing
    if time.time() - t0 < _DEADLINE - 10:
        h_opt = _pgd_optimize(best_h, beta=200.0, steps=400, lr=0.005,
                              deadline=_DEADLINE - 10)
        _update_best(h_opt)

    # Phase 5: Final SLSQP polish
    if time.time() - t0 < _DEADLINE - 3:
        h_opt = _run_slsqp(best_h, beta=500.0, maxiter=150, ftol=1e-11,
                           deadline=_DEADLINE - 2)
        _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h

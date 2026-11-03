"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: Replaces SLSQP with Projected Gradient Descent (PGD) using a
LogSumExp smooth maximum surrogate and exact projection onto the simplex-box
constraint set {0<=h_i<=1, sum(h)=N/2}. This allows far more iterations within
the time budget and avoids SLSQP's quadratic programming overhead.
Refinement machinery: multi-phase annealing (increasing beta), multiple structured
starts, and a final high-beta polish phase.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (LogSumExp).
    Returns (value, gradient).
    Uses FFT-based vectorized gradient computation.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max

    val = m + np.log(sum_exp) / beta

    # grad_j = -(2/n) * sum_k w_k * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    # Term A: sum_k w_k * h_{j-k} for j>=k  -> linear cross-correlation (w * h)[j]
    # Term B: sum_k w_k * h_{j+k} for j+k<n -> linear cross-correlation (h * w)[j]
    
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    
    # TermA[j] = sum_{k=0}^{j} w[k] * h[j-k]
    termA_full = np.fft.irfft(W * np.conj(H), n=n_fft)
    termA = termA_full[:n]
    
    # TermB[j] = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    termB_full = np.fft.irfft(np.conj(W) * H, n=n_fft)
    termB = termB_full[:n]
    
    grad = -(2.0 / n) * (termA + termB)
    
    return val, grad


def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses binary search on the Lagrange multiplier tau.
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


def _pgd_optimize(h0, beta, steps, lr, time_budget):
    """
    Projected Gradient Descent with adaptive learning rate and backtracking.
    Returns the optimized h.
    """
    h = h0.copy()
    target_sum = N / 2.0
    current_lr = lr
    prev_val, _ = _smooth_max_grad(h, beta)
    
    for i in range(steps):
        if time.time() > time_budget:
            break

        val, grad = _smooth_max_grad(h, beta)
        
        # Backtracking line search
        step_lr = current_lr
        new_h = h - step_lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _smooth_max_grad(new_h_proj, beta)
        
        while new_val > prev_val and step_lr > 1e-10:
            step_lr /= 2.0
            new_h = h - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)
        
        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            # Gentle learning rate increase if progress is consistent
            if i % 20 == 0:
                current_lr = min(current_lr * 1.02, 0.05)
        else:
            current_lr = max(current_lr * 0.5, 1e-6)

    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)

    # Generate diverse starting points
    starts = []
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _project_sum_box(h_cos, N / 2.0)
    starts.append(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _project_sum_box(h_sin, N / 2.0)
    starts.append(h_sin)
    
    # Triangle wave
    h_tri = 0.5 + 0.5 * np.abs(2.0 * ((i / N) % 1.0) - 1.0)
    h_tri = _project_sum_box(h_tri, N / 2.0)
    starts.append(h_tri)
    
    # Random starts
    for _ in range(6):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        starts.append(h_rand)
    
    # Evaluate all starts
    for h_s in starts:
        _update_best(h_s)

    if best_h is None:
        best_h = np.full(N, 0.5)
        _update_best(best_h)

    # Multi-phase PGD optimization with increasing beta (annealing)
    # Phase 1: Low beta, coarse optimization from best start
    if time.time() - t0 < _DEADLINE - 50:
        h_opt = _pgd_optimize(best_h, beta=10.0, steps=400, lr=0.05,
                              time_budget=_DEADLINE - 45)
        _update_best(h_opt)

    # Phase 2: Medium beta
    if time.time() - t0 < _DEADLINE - 35:
        h_opt = _pgd_optimize(best_h, beta=30.0, steps=500, lr=0.02,
                              time_budget=_DEADLINE - 30)
        _update_best(h_opt)

    # Phase 3: High beta
    if time.time() - t0 < _DEADLINE - 20:
        h_opt = _pgd_optimize(best_h, beta=80.0, steps=500, lr=0.008,
                              time_budget=_DEADLINE - 15)
        _update_best(h_opt)

    # Phase 4: Very high beta polish
    if time.time() - t0 < _DEADLINE - 8:
        h_opt = _pgd_optimize(best_h, beta=200.0, steps=400, lr=0.003,
                              time_budget=_DEADLINE - 4)
        _update_best(h_opt)

    # Phase 5: Ultra-high beta final polish
    if time.time() - t0 < _DEADLINE - 2:
        h_opt = _pgd_optimize(best_h, beta=500.0, steps=200, lr=0.001,
                              time_budget=_DEADLINE - 1)
        _update_best(h_opt)

    # Final projection to ensure exact constraints
    best_h = _project_sum_box(best_h, N / 2.0)
    return best_h

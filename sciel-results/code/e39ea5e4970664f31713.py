"""Optimize nonnegative step heights to minimize normalized autoconvolution peak."""
import numpy as np
import time

N = 600

def _autoconv_linear(h):
    """Linear autoconvolution of h (length 2N-1)."""
    size = 2 * N
    F = np.fft.rfft(h, size)
    ac = np.fft.irfft(F * F, size)[:2 * N - 1]
    return ac

def _score(h):
    """Normalized maximum of linear autoconvolution."""
    if h is None:
        return float('inf')
    h = np.asarray(h, dtype=float)
    if h.shape[0] != N:
        return float('inf')
    if np.any(h < -1e-12):
        return float('inf')
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s <= 0:
        return float('inf')
    ac = _autoconv_linear(h)
    peak = np.max(ac)
    if not np.isfinite(peak):
        return float('inf')
    return peak / (s * s)

def _soft_max(ac, beta):
    """Log-sum-exp smooth max of ac, returning (value, weights)."""
    m = np.max(ac)
    if not np.isfinite(m):
        return float('inf'), np.zeros_like(ac)
    z = beta * (ac - m)
    z = np.clip(z, -50, 50)
    e = np.exp(z)
    se = np.sum(e)
    if se <= 0:
        se = 1.0
    w = e / se
    val = m + np.log(se) / beta
    return val, w

def _project(h):
    """Project to nonnegativity and normalize sum to 1."""
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s <= 0:
        return np.ones(N) / N
    return h / s

def _grad_smooth_max(h, beta):
    """
    Gradient of smooth_max(autoconv_linear(h)) w.r.t. h.
    Uses FFT to compute the weighted sum of shifted copies of h.
    """
    ac = _autoconv_linear(h)
    _, w = _soft_max(ac, beta)
    # ac[k] = sum_j h[j] h[j-k]  (linear, k in 0..2N-2)
    # d ac[k] / d h[i] = h[i-k] + h[i+k] (with out-of-range terms = 0)
    # grad[i] = sum_k w[k] * (h[i-k] + h[i+k])
    # = (w * h)[i] (linear conv) + (w_rev * h)[i]
    # We want sum_k w[k] h[i-k] = (w * h)[i] linear conv
    # and sum_k w[k] h[i+k] = (w_rev * h)[i] where w_rev[k]=w[-k]
    # For linear conv, use FFT.
    size = 2 * N
    W = np.fft.rfft(w, size)
    H = np.fft.rfft(h, size)
    # conv1[i] = sum_k w[k] h[i-k]
    conv1 = np.fft.irfft(W * H, size)[:2 * N - 1]
    # conv2[i] = sum_k w[k] h[i+k] = sum_k w[k] h[-(i-k)] = (w * h_flipped)
    hf = h[::-1]
    Hf = np.fft.rfft(hf, size)
    conv2 = np.fft.irfft(W * Hf, size)[:2 * N - 1]
    conv2 = conv2[::-1]
    grad = conv1 + conv2
    return grad

def _pgd_optimize(h0, max_iters, deadline, beta_start=30.0, beta_end=200.0,
                  lr_start=0.02, lr_end=0.0005):
    """Projected gradient descent on smooth max of autoconv."""
    h = _project(h0.copy())
    best_h = h.copy()
    best_score = _score(h)
    t0 = time.time()
    for it in range(max_iters):
        if time.time() > deadline:
            break
        beta = beta_start + (beta_end - beta_start) * (it / max(1, max_iters - 1))
        lr = lr_start * (lr_end / lr_start) ** (it / max(1, max_iters - 1))
        grad = _grad_smooth_max(h, beta)
        # Step in direction of minimizing smooth max, but we also want to keep
        # h nonneg and sum=1. Use projected gradient.
        h_new = h - lr * grad
        h_new = _project(h_new)
        # Line search: accept if score improves or smooth max decreases
        s_new = _score(h_new)
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
        h = h_new
    return best_h, best_score

def _make_gaussian(sigma_frac):
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = np.exp(-0.5 * (x / sigma_frac) ** 2)
    return _project(h)

def _make_box(width_frac):
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = np.where(np.abs(x) <= width_frac, 1.0, 0.0)
    return _project(h)

def _make_triangular(width_frac):
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = np.maximum(1 - np.abs(x) / width_frac, 0)
    return _project(h)

def _make_bump(p, width_frac):
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    u = np.abs(x) / width_frac
    h = np.where(u < 1, (1 - u ** 2) ** p, 0)
    return _project(h)

def _make_flat():
    return np.ones(N) / N

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_score = float('inf')
    best_h = _make_flat()

    # Build a diverse set of initial candidates
    candidates = []
    candidates.append(_make_flat())
    for w in [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25]:
        candidates.append(_make_box(w))
    for sig in [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]:
        candidates.append(_make_gaussian(sig))
    for w in [0.05, 0.1, 0.15, 0.2, 0.25]:
        candidates.append(_make_triangular(w))
    for p in [2, 3, 4]:
        for w in [0.1, 0.15, 0.2, 0.25]:
            candidates.append(_make_bump(p, w))

    # Evaluate all candidates
    for c in candidates:
        if time.time() > deadline:
            break
        s = _score(c)
        if s < best_score:
            best_score = s
            best_h = c.copy()

    # Run projected gradient descent from the best candidate and a few others
    # Allocate time per optimization run
    runs = [best_h]
    # Add a few diverse starting points
    for sig in [0.05, 0.1, 0.15]:
        runs.append(_make_gaussian(sig))
    for w in [0.1, 0.2]:
        runs.append(_make_box(w))

    # Time budget per run
    elapsed = time.time() - start_time
    remaining = max(0.0, deadline - elapsed - 5.0)
    per_run = remaining / max(1, len(runs))

    for i, h0 in enumerate(runs):
        if time.time() > deadline - 5:
            break
        run_deadline = min(deadline - 5.0, time.time() + per_run)
        max_iters = int(max(10, per_run * 20))  # rough estimate
        h_opt, s_opt = _pgd_optimize(h0, max_iters, run_deadline,
                                     beta_start=30.0, beta_end=200.0,
                                     lr_start=0.02, lr_end=0.0005)
        if s_opt < best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Final local refinement with small perturbations
    if time.time() < deadline - 5 and best_score < float('inf'):
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            noise = np.random.randn(N)
            kernel = np.exp(-np.arange(-15, 16) ** 2 / 8)
            noise = np.convolve(noise, kernel, mode='same')
            noise = (noise - noise.mean()) * 0.003
            h_pert = best_h + noise
            h_pert = _project(h_pert)
            s = _score(h_pert)
            if s < best_score:
                best_score = s
                best_h = h_pert.copy()

    # Final safety check
    best_h = np.maximum(best_h, 0.0)
    s = best_h.sum()
    if s <= 0:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s

    return list(best_h)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END

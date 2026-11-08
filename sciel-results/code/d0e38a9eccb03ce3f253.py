"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 150.0  # wall-clock limit
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _elapsed():
    return time.time() - _START_TIME

_START_TIME = time.time()

def _compute_ratio(h):
    """Compute (L2^2) / (L1 * sup(autoconv)) for a nonnegative height sequence."""
    h = np.asarray(h, dtype=np.float64)
    if h.min() < 0:
        h = np.clip(h, 0.0, None)
    l2_sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return 0.0
    # Autoconvolution: (h * h) via FFT for efficiency
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    # The autoconvolution of a sequence of length N has length 2N-1
    conv = conv[:2 * N - 1]
    sup_conv = float(np.max(conv))
    if sup_conv <= 1e-300:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _ratio_and_grad(params):
    """Compute ratio and gradient with respect to Gaussian parameters.
    
    params = [center, sigma, amplitude, floor]
    h_i = amplitude * exp(-((i - center)/sigma)^2) + floor
    """
    center, sigma, amplitude, floor = params
    sigma = max(abs(sigma), 1e-6)
    i = np.arange(N, dtype=np.float64)
    h = amplitude * np.exp(-((i - center) / sigma) ** 2) + floor
    h = np.clip(h, 0.0, None)
    
    ratio = _compute_ratio(h)
    
    # Numerical gradient (central differences) for robustness
    grad = np.zeros(4)
    step = 1e-6
    for k in range(4):
        p_plus = params.copy()
        p_plus[k] += step
        p_minus = params.copy()
        p_minus[k] -= step
        grad[k] = (_compute_ratio_from_params(p_plus) - _compute_ratio_from_params(p_minus)) / (2 * step)
    
    return -ratio, grad  # Negate for minimization

def _compute_ratio_from_params(params):
    center, sigma, amplitude, floor = params
    sigma = max(abs(sigma), 1e-6)
    i = np.arange(N, dtype=np.float64)
    h = amplitude * np.exp(-((i - center) / sigma) ** 2) + floor
    h = np.clip(h, 0.0, None)
    return _compute_ratio(h)

def _direct_optimize_h(h0):
    """Optimize heights directly using L-BFGS-B with numerical gradients."""
    def neg_ratio(h):
        return -_compute_ratio(h)
    
    # Use finite difference gradient
    res = minimize(
        neg_ratio,
        h0,
        method='L-BFGS-B',
        bounds=[(0.0, None)] * N,
        options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    return res.x, -res.fun

def _polish(h):
    """High-precision polish pass with tighter tolerances."""
    def neg_ratio(h):
        return -_compute_ratio(h)
    
    res = minimize(
        neg_ratio,
        h,
        method='L-BFGS-B',
        bounds=[(0.0, None)] * N,
        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
    )
    return res.x, -res.fun

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_h = np.ones(N)
    best_ratio = _compute_ratio(best_h)
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform (baseline)
    candidates.append(np.ones(N))
    
    # 2. Various Gaussian initializations
    for center in [N // 2, N // 2 - 2, N // 2 + 2]:
        for sigma in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
            i = np.arange(N, dtype=np.float64)
            h = np.exp(-((i - center) / sigma) ** 2)
            candidates.append(h)
    
    # 3. Triangular / tent
    i = np.arange(N, dtype=np.float64)
    center = N // 2
    h_tent = np.maximum(0, 1 - np.abs(i - center) / (N // 2))
    candidates.append(h_tent)
    
    # 4. Raised cosine
    theta = 2 * np.pi * i / (N - 1)
    h_cos = 0.5 * (1 + np.cos(theta))
    candidates.append(h_cos)
    
    # 5. Parabolic
    i_norm = (i - center) / (N / 2)
    h_para = np.maximum(0, 1 - i_norm ** 2)
    candidates.append(h_para)
    
    # 6. Multi-peak: two Gaussians
    for d in [3, 5, 7]:
        h_two = np.exp(-((i - (center - d)) / 3.0) ** 2) + np.exp(-((i - (center + d)) / 3.0) ** 2)
        candidates.append(h_two)
    
    # Evaluate all candidates
    for h in candidates:
        r = _compute_ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Check deadline before optimization
    if _elapsed() > DEADLINE - SAFETY_MARGIN:
        return list(best_h)
    
    # Optimize the top few candidates
    scored = [(_compute_ratio(h), h) for h in candidates]
    scored.sort(key=lambda x: -x[0])
    top_candidates = [h for _, h in scored[:5]]
    
    for h0 in top_candidates:
        if _elapsed() > DEADLINE - SAFETY_MARGIN:
            break
        try:
            h_opt, ratio = _direct_optimize_h(h0)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish if time permits
    if _elapsed() < DEADLINE - SAFETY_MARGIN - 10:
        try:
            h_polished, ratio = _polish(best_h)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # Ensure non-negativity
    best_h = np.clip(best_h, 0.0, None)
    
    return list(best_h)

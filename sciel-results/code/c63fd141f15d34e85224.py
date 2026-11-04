import time
import numpy as np

N = 50
w = 0.5 / N  # step width = 0.01
_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _autoconv_lags(h):
    """Compute all lag values of autoconvolution (without the w factor)."""
    c = np.zeros(N)
    c[0] = np.dot(h, h)
    for m in range(1, N):
        c[m] = np.dot(h[:N - m], h[m:])
    return c


def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) using hard max."""
    h = np.maximum(h, 0.0)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    c = _autoconv_lags(h)
    acmax = w * c.max()
    if acmax <= 0:
        return 0.0
    return s2 / (s1 * acmax)


def _soft_max(c, beta):
    """Soft max of c using log-sum-exp trick."""
    cmax = c.max()
    if cmax <= 0:
        return 0.0
    return cmax + np.log(np.sum(np.exp((c - cmax) / beta)) * beta)


def _ratio_smooth(h, beta):
    """Smoothed ratio for gradient-based optimization."""
    h = np.maximum(h, 1e-12)
    s2 = np.dot(h, h)
    s1 = np.sum(h)
    if s1 <= 0 or s2 <= 0:
        return 0.0
    c = _autoconv_lags(h)
    acmax_soft = w * _soft_max(c, beta)
    if acmax_soft <= 0:
        return 0.0
    return s2 / (s1 * acmax_soft)


def _ratio_and_grad(h, beta=10.0):
    """Compute ratio and gradient via finite differences."""
    h = np.maximum(h, 1e-12)
    f0 = _ratio_smooth(h, beta)
    grad = np.zeros(N)
    eps = 1e-6
    for i in range(N):
        h[i] += eps
        f1 = _ratio_smooth(h, beta)
        h[i] -= eps
        grad[i] = (f1 - f0) / eps
    return f0, grad


def _optimize_gradient(h0, max_iters=500, lr_init=0.1, beta=10.0):
    """Gradient ascent with adaptive learning rate."""
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 1e-12)
    best_h = h.copy()
    best_f = _ratio_smooth(h, beta)
    lr = lr_init
    
    for it in range(max_iters):
        if _elapsed() > DEADLINE - 5:
            break
        
        f, grad = _ratio_and_grad(h, beta)
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            step = grad / grad_norm * lr
            h_new = np.maximum(h + step, 0.0)
            f_new = _ratio_smooth(h_new, beta)
            
            if f_new > best_f:
                best_f = f_new
                best_h = h_new.copy()
                h = h_new
                lr = min(lr * 1.1, 0.5)
            else:
                lr *= 0.7
        else:
            break
    
    return best_h, best_f


def _gaussian_profile(center, sigma, amplitude=1.0):
    """Create a Gaussian profile centered at 'center' with width sigma."""
    x = np.arange(N)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _lorentzian_profile(center, gamma, amplitude=1.0):
    """Create a Lorentzian profile."""
    x = np.arange(N)
    return amplitude / (1 + ((x - center) / gamma) ** 2)


def _random_profile(rng):
    """Generate a random smooth profile."""
    kind = rng.integers(0, 3)
    if kind == 0:
        center = rng.uniform(0, N)
        sigma = rng.uniform(2, 10)
        amp = rng.uniform(0.5, 2.0)
        return _gaussian_profile(center, sigma, amp)
    elif kind == 1:
        center = rng.uniform(0, N)
        gamma = rng.uniform(2, 8)
        amp = rng.uniform(0.5, 2.0)
        return _lorentzian_profile(center, gamma, amp)
    else:
        # Sum of two Gaussians
        c1 = rng.uniform(0, N)
        c2 = rng.uniform(0, N)
        s1 = rng.uniform(2, 6)
        s2 = rng.uniform(2, 6)
        a1 = rng.uniform(0.3, 1.0)
        a2 = rng.uniform(0.3, 1.0)
        return _gaussian_profile(c1, s1, a1) + _gaussian_profile(c2, s2, a2)


def construct():
    best_h = np.ones(N)
    best_f = _ratio(best_h)
    
    # Phase 1: Try various smooth profiles
    rng = np.random.default_rng(42)
    for _ in range(100):
        if _elapsed() > DEADLINE - 30:
            break
        h = _random_profile(rng)
        h = np.maximum(h, 0.0)
        f = _ratio(h)
        if f > best_f:
            best_f = f
            best_h = h.copy()
    
    # Phase 2: Gradient-based optimization from best profile
    if _elapsed() < DEADLINE - 20:
        for beta in [5.0, 10.0, 20.0]:
            if _elapsed() > DEADLINE - 10:
                break
            h_opt, f_opt = _optimize_gradient(best_h, max_iters=300, lr_init=0.05, beta=beta)
            f_hard = _ratio(h_opt)
            if f_hard > best_f:
                best_f = f_hard
                best_h = h_opt.copy()
    
    # Phase 3: Try uniform and simple patterns
    simple_patterns = [
        np.ones(N),
        np.linspace(1, 0.1, N),
        np.linspace(0.1, 1, N),
        np.sin(np.pi * np.arange(N) / N) + 0.1,
        np.cos(np.pi * np.arange(N) / N) ** 2 + 0.1,
    ]
    for pat in simple_patterns:
        pat = np.maximum(pat, 0.0)
        f = _ratio(pat)
        if f > best_f:
            best_f = f
            best_h = pat.copy()
    
    # Phase 4: Final refinement with gradient optimization
    if _elapsed() < DEADLINE - 5:
        h_opt, f_opt = _optimize_gradient(best_h, max_iters=200, lr_init=0.02, beta=15.0)
        f_hard = _ratio(h_opt)
        if f_hard > best_f:
            best_f = f_hard
            best_h = h_opt.copy()
    
    best_h = np.maximum(best_h, 0.0)
    return [float(v) for v in best_h]

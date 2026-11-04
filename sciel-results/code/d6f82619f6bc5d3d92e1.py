import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _soft_max_abs(c, beta):
    """Compute soft-max of absolute values."""
    abs_c = np.abs(c)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0
    exps = np.exp(beta * (abs_c - m))
    sum_exps = np.sum(exps)
    return m + np.log(sum_exps) / beta

def _objective_smooth(x, beta):
    """Smooth objective: soft-max |c_k| / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv_lin(x)
    sm = _soft_max_abs(c, beta)
    return sm / (s * s)

def _objective_grad(x, beta):
    """Objective and gradient for L-BFGS."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(x)
    
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    c = _autoconv_lin(x)
    abs_c = np.abs(c)
    m = np.max(abs_c)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    exps = np.exp(beta * (abs_c - m))
    sum_exps = np.sum(exps)
    weights = exps / sum_exps
    
    # Objective value
    sm = m + np.log(sum_exps) / beta
    obj = sm / (s * s)
    
    # Gradient computation using FFT
    # d(sm)/dx = (1/s^2) * d(sm)/dx - (2*s/s^2) * sm * (1/s) ... wait
    # Let f = sm / s^2
    # df/dx_j = (1/s^2) * dsm/dx_j - 2*sm/s^3 * (ds/dx_j)
    # ds/dx_j = 1
    # dsm/dx_j = sum_k weights[k] * sign(c[k]) * (x[j-k] + x[j+k])
    
    g = weights * np.sign(c)
    g_padded = np.zeros(size)
    g_padded[:len(g)] = g
    G = np.fft.rfft(g_padded, size)
    X = np.fft.rfft(x, size)
    dsm_dx = np.fft.irfft(2 * X * G, size)[:n]
    
    grad = (1.0 / (s * s)) * dsm_dx - (2.0 * sm / (s ** 3)) * 1.0
    
    return obj, grad

def _generate_random_sign(rng):
    x = rng.choice([-1.0, 1.0], size=N)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 1.0
        s = np.sum(x)
    return x / s

def _generate_random_gauss(rng):
    x = rng.normal(0, 1, size=N)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 1.0
        s = np.sum(x)
    return x / s

def _generate_multi_sine(rng, n_frequencies=5):
    """Generate a sum of random-phase sines."""
    x = np.zeros(N)
    for _ in range(n_frequencies):
        freq = rng.uniform(0.5, N / 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.0)
        t = np.arange(N)
        x += amp * np.sin(2 * np.pi * freq * t / N + phase)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 1.0
        s = np.sum(x)
    return x / s

def _generate_chirp(rng):
    """Generate a chirp signal."""
    t = np.arange(N)
    f0 = rng.uniform(0.1, 1.0)
    f1 = rng.uniform(2.0, N / 4.0)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * N))
    x = np.sin(phase)
    s = np.sum(x)
    if abs(s) < 1e-6:
        x += 1.0
        s = np.sum(x)
    return x / s

def _refine_lbfgs(x0, deadline, beta_start=20.0, beta_end=100.0, max_iter=50):
    """Refine using L-BFGS-B with annealed beta."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_std(x)
    
    # Use a moderate beta for L-BFGS
    beta = beta_start
    
    def obj_func(x):
        if time.time() > deadline:
            return 1e10
        return _objective_smooth(x, beta)
    
    def obj_grad_func(x):
        if time.time() > deadline:
            return 1e10, np.zeros_like(x)
        return _objective_grad(x, beta)
    
    try:
        result = minimize(
            obj_grad_func, x, method='L-BFGS-B', jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100}
        )
        if result.x is not None:
            x_new = result.x
            s_new = np.sum(x_new)
            if abs(s_new) > 1e-12:
                x_new = x_new / s_new
            val = _metric_std(x_new)
            if val < best_val:
                best_val = val
                best_x = x_new.copy()
    except Exception:
        pass
    
    return best_x

def construct():
    deadline = time.time() + 130.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_x = np.ones(N)
    best_val = _metric_std(best_x)
    
    # Generate diverse initial candidates
    candidates = []
    
    # Random signs
    for i in range(3):
        candidates.append(_generate_random_sign(rng))
    
    # Random Gaussian
    for i in range(3):
        candidates.append(_generate_random_gauss(rng))
    
    # Multi-sine
    for i in range(3):
        candidates.append(_generate_multi_sine(rng, n_frequencies=4 + i))
    
    # Chirp
    for i in range(2):
        candidates.append(_generate_chirp(rng))
    
    # Evaluate initial metrics
    for x_cand in candidates:
        val = _metric_std(x_cand)
        if val < best_val:
            best_val = val
            best_x = x_cand.copy()
    
    # Sort candidates by initial metric (best first)
    candidates.sort(key=lambda x: _metric_std(x))
    
    # Refine with L-BFGS, prioritizing best candidates
    for x_cand in candidates:
        if time.time() > deadline - 5:
            break
        
        x_opt = _refine_lbfgs(x_cand, deadline=deadline - 2)
        val = _metric_std(x_opt)
        
        if val < best_val:
            best_val = val
            best_x = x_opt.copy()
    
    # If time remains, do additional random restarts with refinement
    while time.time() < deadline - 10:
        x_init = _generate_random_gauss(rng)
        x_opt = _refine_lbfgs(x_init, deadline=deadline - 2, max_iter=30)
        val = _metric_std(x_opt)
        
        if val < best_val:
            best_val = val
            best_x = x_opt.copy()
    
    return best_x

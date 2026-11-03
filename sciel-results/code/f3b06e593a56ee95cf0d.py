import numpy as np
import time
from scipy.optimize import minimize

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
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    Vectorized gradient computation using FFT.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Compute gradient using FFT-based convolution
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # conv1[j] = sum_k w[k] * h[j-k] (linear convolution)
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # conv2[j] = sum_k w[k] * h[j+k]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    # Combine
    grad = -(2.0 / n) * (conv1 + conv2)
    
    return val, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
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


def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t subject to cs <= t, sum(h) = N/2, 0 <= h <= 1.
    x = [h_0, ..., h_{N-1}, t]
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    
    # Smooth constraint violation: sum of max(0, cs - t)^2
    violations = np.maximum(cs - t, 0)
    penalty = np.sum(violations ** 2)
    
    # Smooth max approximation for better gradient
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    soft_max = m + np.log(sum_exp) / beta
    
    # Objective: t + penalty + small regularization
    return t + penalty + 1e-6 * (t - soft_max) ** 2


def _epigraph_gradient(x, beta):
    """
    Gradient of epigraph objective.
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    
    # Gradient of penalty term
    violations = np.maximum(cs - t, 0)
    active = violations > 0
    
    # d/dh of penalty: -2 * sum_{k: cs_k > t} (cs_k - t) * dcs_k/dh
    # dcs_k/dh_j = -(2/N) * (h_{j-k} + h_{j+k}) for valid indices
    n = N
    n_fft = 2 * n
    w = np.where(active, cs - t, 0)
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    grad_h = -2.0 * (-(2.0 / n) * (conv1 + conv2))
    
    # Gradient of t term: 1 + 2*1e-6*(t - soft_max)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    soft_max = m + np.log(sum_exp) / beta
    grad_t = 1.0 + 2e-6 * (t - soft_max)
    
    # Add penalty gradient contribution to grad_t
    grad_t += 2.0 * np.sum(violations)
    
    grad = np.zeros(N + 1)
    grad[:N] = grad_h
    grad[N] = grad_t
    
    return grad


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Analytical Start: Cosine Wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Multi-stage optimization with adaptive beta
    def _run_optimization(h0, beta, maxiter, ftol, method='SLSQP'):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _smooth_max_grad(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _smooth_max_grad(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method=method,
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    def _run_epigraph_optimization(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
        # Initial t estimate
        cs = _linear_overlap_values(h0)
        t0_init = np.max(cs) + 0.1
        
        x0 = np.concatenate([h0, [t0_init]])
        bounds = [(0.0, 1.0)] * N + [(0.0, 2.0)]
        
        def obj_func(x):
            return _epigraph_objective(x, beta)
            
        def jac_func(x):
            return _epigraph_gradient(x, beta)
            
        try:
            res = minimize(obj_func, x0.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or _epigraph_objective(res.x, beta) < best_score:
                h_res = _fix_sum(res.x[:N])
                _update_best(h_res)
        except Exception:
            pass

    # Phase 1: Coarse optimization with low beta
    if time.time() - t0 < _DEADLINE - 60:
        _run_optimization(h_cos, beta=20.0, maxiter=100, ftol=1e-6)
        
    # Phase 2: Medium optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 40:
        _run_optimization(best_h, beta=50.0, maxiter=150, ftol=1e-8)
        
    # Phase 3: Try epigraph formulation
    if best_h is not None and time.time() - t0 < _DEADLINE - 25:
        _run_epigraph_optimization(best_h, beta=50.0, maxiter=100, ftol=1e-8)
        
    # Phase 4: Fine optimization with high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        _run_optimization(best_h, beta=100.0, maxiter=200, ftol=1e-10)
        
    # Phase 5: Final polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_optimization(best_h, beta=200.0, maxiter=300, ftol=1e-12)
        
    # Phase 6: Try a few random restarts if time permits
    if time.time() - t0 < _DEADLINE - 20:
        for _ in range(2):
            if time.time() - t0 > _DEADLINE - 10:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_optimization(h_rand, beta=50.0, maxiter=80, ftol=1e-8)
            if best_h is not None and time.time() - t0 < _DEADLINE - 10:
                _run_optimization(best_h, beta=100.0, maxiter=150, ftol=1e-10)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h

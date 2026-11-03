import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 100
_DEADLINE = time.time() + 135  # Safety margin before 150s kill

def _exact_grad(h, k):
    """Compute exact gradient of C_k with respect to h."""
    n = len(h)
    grad = np.zeros(n)
    factor = -(2.0 / n)
    if k > 0:
        grad[k:] += factor * h[:n-k]
    if k < n:
        grad[:n-k] += factor * h[k:]
    return grad

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
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

def _logsumexp_obj_and_grad(h, beta):
    """
    Compute the LogSumExp smoothed maximum of C_k and its gradient.
    Objective: F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    Gradient: sum(w_k * grad_C_k)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Avoid overflow
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # Only compute gradients for significant weights
    significant = w > 1e-10
    for k in np.where(significant)[0]:
        wk = w[k]
        if wk > 0:
            grad += wk * _exact_grad(h, k)
            
    return val, grad

def _slsqp_optimize(h0, beta, max_iter=1000):
    """
    Optimize using SLSQP with explicit sum constraint.
    """
    if time.time() > _DEADLINE - 5:
        return h0
        
    # Cache to avoid recomputing gradient when fun and jac are called with same h
    cache = {'h': None, 'val': None, 'grad': None}
    
    def fun(h):
        h_arr = np.asarray(h, dtype=np.float64)
        if cache['h'] is not None and np.array_equal(h_arr, cache['h']):
            return cache['val']
        val, grad = _logsumexp_obj_and_grad(h_arr, beta)
        cache['h'] = h_arr.copy()
        cache['val'] = val
        cache['grad'] = grad
        return val
        
    def jac(h):
        h_arr = np.asarray(h, dtype=np.float64)
        if cache['h'] is not None and np.array_equal(h_arr, cache['h']):
            return cache['grad']
        val, grad = _logsumexp_obj_and_grad(h_arr, beta)
        cache['h'] = h_arr.copy()
        cache['val'] = val
        cache['grad'] = grad
        return grad
        
    # Constraints
    constraints = [
        {
            'type': 'eq',
            'fun': lambda h: np.sum(h) - N / 2.0,
            'jac': lambda h: np.ones(N)
        }
    ]
    
    bounds = [(0.0, 1.0)] * N
    
    try:
        res = minimize(
            fun, 
            h0, 
            method='SLSQP', 
            jac=jac, 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        h_new = res.x
        # Project to ensure feasibility
        h_new = _project_sum_box(h_new, N / 2.0)
        return h_new
    except Exception:
        return h0

def _exact_max_polish(h0, max_iter=500):
    """
    Polish by directly minimizing the max C_k using a projected subgradient method
    with adaptive step size, focusing on the active constraints.
    """
    if time.time() > _DEADLINE - 3:
        return h0
    
    h = h0.copy()
    best_h = h.copy()
    best_score = _score(h)
    
    step_size = 0.01
    momentum = np.zeros(N)
    beta_mom = 0.9
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 3:
            break
        
        cs = _linear_overlap_values(h)
        max_c = np.max(cs)
        
        # Find active constraints (near max)
        threshold = max_c - 0.001
        active = cs >= threshold
        
        if not np.any(active):
            active = np.array([np.argmax(cs)])
        
        # Subgradient of max C_k at active k
        grad = np.zeros(N)
        for k in np.where(active)[0]:
            grad += _exact_grad(h, k)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Projected gradient step
        momentum = beta_mom * momentum + (1 - beta_mom) * grad / grad_norm
        h_new = h - step_size * momentum
        
        # Project to feasible set
        h_new = _project_sum_box(h_new, N / 2.0)
        
        score_new = _score(h_new)
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
        
        # Adaptive step size
        if score_new > max_c + 0.001:
            step_size *= 0.7
        else:
            step_size = min(step_size * 1.05, 0.1)
        
        h = h_new
    
    return best_h

def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
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
        
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        # Ensure strict feasibility
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _project_sum_box(h_cos, N / 2.0)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _project_sum_box(h_sin, N / 2.0)
    _update_best(h_sin)
    
    # Random starts
    starts = [h_cos, h_sin]
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        starts.append(h_rand)
        _update_best(h_rand)
        
    # 2. Optimization Phases using SLSQP
    # We refine the best candidate found so far, and potentially others if time permits.
    # Strategy: Take the current best_h and refine it with increasing beta.
    
    # Phase 1: Coarse optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 60:
        h_opt = _slsqp_optimize(best_h, beta=10.0, max_iter=200)
        _update_best(h_opt)
        
    # Phase 2: Medium beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 40:
        h_opt = _slsqp_optimize(best_h, beta=50.0, max_iter=400)
        _update_best(h_opt)
        
    # Phase 3: High beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 25:
        h_opt = _slsqp_optimize(best_h, beta=100.0, max_iter=600)
        _update_best(h_opt)
        
    # Phase 4: Ultra-high beta polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_opt = _slsqp_optimize(best_h, beta=200.0, max_iter=1000)
        _update_best(h_opt)
        
    # Phase 5: Very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt = _slsqp_optimize(best_h, beta=500.0, max_iter=1500)
        _update_best(h_opt)
        
    # Phase 6: Exact max polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_opt = _exact_max_polish(best_h, max_iter=500)
        _update_best(h_opt)
        
    # Try alternative starts if time permits
    if time.time() - t0 < _DEADLINE - 20:
        # Try a few more structured starts
        for freq in [0.7, 1.3, 2.0]:
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_alt = 0.5 + 0.4 * np.cos(2.0 * np.pi * freq * i / N + np.pi/4)
            h_alt = _project_sum_box(h_alt, N / 2.0)
            h_opt = _slsqp_optimize(h_alt, beta=100.0, max_iter=400)
            _update_best(h_opt)
            if time.time() - t0 < _DEADLINE - 10:
                h_opt = _slsqp_optimize(h_opt, beta=300.0, max_iter=800)
                _update_best(h_opt)
            if time.time() - t0 < _DEADLINE - 8:
                h_opt = _exact_max_polish(h_opt, max_iter=300)
                _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final projection to ensure exact sum and bounds
    best_h = _project_sum_box(best_h, N / 2.0)
    
    return best_h

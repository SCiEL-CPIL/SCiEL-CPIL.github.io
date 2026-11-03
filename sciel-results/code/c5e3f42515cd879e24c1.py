import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # 10s safety margin before 150s kill

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
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    shifted = beta * (cs - m)
    shifted = np.clip(shifted, -500, 500)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * dR_k/dh_j
    # Vectorized gradient computation using FFT-based correlation logic
    # For each k, contribution to grad[j] from C_k is:
    # if j >= k: -w[k]*(2/n)*h[j-k]
    # if j+k < n: -w[k]*(2/n)*h[j+k]
    
    # We can compute this by summing over k.
    # grad[j] = -(2/n) * sum_k w[k] * (h[j-k] (if j>=k) + h[j+k] (if j+k<n))
    
    # Let's stick to the loop for clarity and correctness, it's fast enough for N=100
    # unless we optimize heavily.
    
    for k in range(n):
        if w[k] < 1e-10:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

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
        
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)

def _nesterov_pgd(h0, beta, steps, lr, deadline):
    """
    Nesterov-accelerated Projected Gradient Descent.
    """
    h = h0.copy()
    y = h0.copy()
    target_sum = N / 2.0
    prev_val, _ = _smooth_max_grad(h, beta)
    t = 1.0
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        # Nesterov step
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = h + (t / t_new) * (h - y)
        
        val, grad = _smooth_max_grad(y, beta)
        
        # Gradient step from y
        new_h = y - lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _smooth_max_grad(new_h_proj, beta)
        
        # Backtracking if needed
        step_lr = lr
        while new_val > prev_val + 1e-12 and step_lr > 1e-10:
            step_lr /= 2.0
            new_h = y - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)
            
        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            t = t_new
        else:
            # Reset momentum if no progress
            y = h.copy()
            t = 1.0
            if step_lr < 1e-10:
                break
                
    return h

def _refine_with_lbfgs(h0, deadline):
    """
    Refine a solution using L-BFGS-B with the smooth objective.
    """
    h = h0.copy()
    target_sum = N / 2.0
    
    def obj_and_grad(x):
        val, grad = _smooth_max_grad(x, beta=2000.0)
        return val, grad
        
    try:
        res = minimize(
            obj_and_grad,
            h,
            jac=True,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res.fun < _score(h):
            h_new = _project_sum_box(res.x, target_sum)
            return h_new
    except Exception:
        pass
        
    return h

def _coordinate_polish(h0, deadline):
    """
    Coordinate descent polish on the active set of constraints.
    """
    h = h0.copy()
    target_sum = N / 2.0
    current_score = _score(h)
    
    for _ in range(200):
        if time.time() > deadline:
            break
            
        cs = _linear_overlap_values(h)
        max_val = np.max(cs)
        # Only consider a subset of active constraints to save time
        active = np.where(cs > max_val - 0.05)[0]
        
        improved = False
        for j in range(N):
            # Try small perturbations
            for delta in [0.001, -0.001, 0.0005, -0.0005]:
                h_trial = h.copy()
                h_trial[j] += delta
                h_trial = _project_sum_box(h_trial, target_sum)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if time.time() > deadline:
                break
        if not improved:
            break
            
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
            
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Uniform
    h_unif = np.full(N, 0.5)
    _update_best(h_unif)
    
    # Chirps (Quadratic Phase)
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0]:
        for phase in [0, np.pi/2]:
            h_chirp = 0.5 + 0.5 * np.cos(2.0 * np.pi * alpha * (i**2) / N + phase)
            h_chirp = _project_sum_box(h_chirp, N / 2.0)
            _update_best(h_chirp)

    # Linear Frequencies
    for freq in [1, 2, 3, 5, 10]:
        h_lin = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N)
        h_lin = _project_sum_box(h_lin, N / 2.0)
        _update_best(h_lin)
        
    # Random
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)

    # 2. Optimization Phases
    # Phase 1: Coarse Nesterov PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 50:
        h_opt = _nesterov_pgd(best_h, beta=10.0, steps=500, lr=0.05, deadline=_DEADLINE - 50)
        _update_best(h_opt)
        
    # Phase 2: Medium Nesterov PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _nesterov_pgd(best_h, beta=50.0, steps=800, lr=0.01, deadline=_DEADLINE - 30)
        _update_best(h_opt)
        
    # Phase 3: High Nesterov PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_opt = _nesterov_pgd(best_h, beta=200.0, steps=1000, lr=0.005, deadline=_DEADLINE - 15)
        _update_best(h_opt)
        
    # Phase 4: L-BFGS-B Polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_opt = _refine_with_lbfgs(best_h, _DEADLINE - 5)
        _update_best(h_opt)
        
    # Phase 5: Coordinate Descent Polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        h_opt = _coordinate_polish(best_h, _DEADLINE - 1)
        _update_best(h_opt)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_sum_box(best_h, N / 2.0)
    return best_h

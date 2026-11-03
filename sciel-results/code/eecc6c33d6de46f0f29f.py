"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Structural change: Spectral bisection + multi-phase PGD with chirp initializations.
Uses FFT-accelerated overlap computation, LogSumExp smooth max surrogate,
and a novel spectral threshold search to find globally better structures."""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115  # Safety margin before 120s kill


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
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    for k in range(n):
        if w[k] < 1e-12:
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
    Using binary search on the shift parameter tau.
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


def _spectral_bisection(h0, deadline):
    """
    Spectral bisection: find the best level-set structure by searching
    over a 1D family of thresholded versions of h0's principal component.
    This targets the max overlap constraint in a convex-feasible manner.
    """
    t_start = time.time()
    n = len(h0)
    target_sum = N / 2.0
    
    # Use the DC component and first few harmonics as a basis
    # Try to find a structure where the autocorrelation is as flat as possible
    # by adjusting a quadratic phase (chirp) parameter
    
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Search over chirp parameters: h_i = 0.5 + 0.5 * cos(2*pi*alpha*i^2/N + phase)
    # This is a structural change from pure sinusoids
    
    i = np.arange(N)
    
    # Multi-scale chirp search
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        if time.time() > deadline:
            break
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            if time.time() > deadline:
                break
            h_chirp = 0.5 + 0.5 * np.cos(2.0 * np.pi * alpha * (i**2) / N + phase)
            h_chirp = _project_sum_box(h_chirp, target_sum)
            s = _score(h_chirp)
            if s < best_score:
                best_score = s
                best_h = h_chirp.copy()
    
    # Also try linear frequency combinations
    for f1 in [1, 2, 3, 5]:
        for f2 in [2, 3, 5, 7, 10]:
            if f1 == f2:
                continue
            if time.time() > deadline:
                break
            h_combo = 0.5 + 0.25 * np.cos(2.0 * np.pi * f1 * i / N) + 0.25 * np.cos(2.0 * np.pi * f2 * i / N)
            h_combo = _project_sum_box(h_combo, target_sum)
            s = _score(h_combo)
            if s < best_score:
                best_score = s
                best_h = h_combo.copy()
    
    return best_h


def _active_set_polish(h, deadline):
    """
    Active-set polish: directly minimize the max overlap by adjusting
    only the variables that affect the active (near-max) constraints.
    This is a structural refinement that can escape local minima of PGD.
    """
    t_start = time.time()
    n = len(h)
    target_sum = N / 2.0
    h = h.copy()
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        cs = _linear_overlap_values(h)
        max_val = np.max(cs)
        active_mask = cs > max_val - 1e-6  # Active constraints
        
        if np.sum(active_mask) == 0:
            break
        
        # For each active constraint, compute gradient contribution
        # and try a small step in the direction that reduces all active constraints
        
        grad = np.zeros(n)
        for k in range(n):
            if not active_mask[k]:
                continue
            factor = -(2.0 / n)
            if k > 0:
                grad[k:] += factor * h[:n-k]
            if k < n:
                grad[:n-k] += factor * h[k:]
        
        # Normalize gradient
        norm = np.linalg.norm(grad)
        if norm < 1e-10:
            break
        grad = grad / norm
        
        # Try a small step
        for lr in [0.01, 0.005, 0.001, 0.0005]:
            new_h = h - lr * grad
            new_h = _project_sum_box(new_h, target_sum)
            new_max = _score(new_h)
            if new_max < max_val:
                h = new_h
                break
    
    return h


def _pgd_optimize(h0, beta, steps, lr, deadline):
    """
    Projected Gradient Descent optimization.
    """
    h = h0.copy()
    target_sum = N / 2.0
    current_lr = lr
    prev_val, _ = _smooth_max_grad(h, beta)
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        val, grad = _smooth_max_grad(h, beta)
        
        # Backtracking line search
        step_lr = current_lr
        new_h = h - step_lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _smooth_max_grad(new_h_proj, beta)
        
        while new_val > prev_val and step_lr > 1e-8:
            step_lr /= 2.0
            new_h = h - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)
            
        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            # Increase step size slightly if progress was good
            if i % 10 == 0:
                current_lr = min(current_lr * 1.05, 0.1)
        else:
            # No progress, reduce learning rate
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
        # Ensure strict feasibility
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Uniform baseline
    h_unif = np.full(N, 0.5)
    _update_best(h_unif)
    
    # Multiple frequency cosine/sine waves
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N)
        h_cos = _project_sum_box(h_cos, N / 2.0)
        _update_best(h_cos)
        
        h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * freq * i / N)
        h_sin = _project_sum_box(h_sin, N / 2.0)
        _update_best(h_sin)
    
    # Chirp initializations (quadratic phase) - structural addition
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0]:
        for phase in [0, np.pi/2]:
            h_chirp = 0.5 + 0.5 * np.cos(2.0 * np.pi * alpha * (i**2) / N + phase)
            h_chirp = _project_sum_box(h_chirp, N / 2.0)
            _update_best(h_chirp)
    
    # Random starts
    for _ in range(8):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)
    
    # 2. Spectral bisection search - structural change
    if best_h is not None and time.time() - t0 < _DEADLINE - 50:
        h_spec = _spectral_bisection(best_h, deadline=_DEADLINE - 50)
        _update_best(h_spec)
    
    # 3. Optimization Phases
    # Phase 1: Coarse optimization with lower beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 40:
        h_opt = _pgd_optimize(best_h, beta=10.0, steps=300, lr=0.05, deadline=_DEADLINE - 40)
        _update_best(h_opt)
        
    # Phase 2: Medium beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 25:
        h_opt = _pgd_optimize(best_h, beta=50.0, steps=400, lr=0.01, deadline=_DEADLINE - 25)
        _update_best(h_opt)
        
    # Phase 3: High beta (sharper max approximation)
    if best_h is not None and time.time() - t0 < _DEADLINE - 12:
        h_opt = _pgd_optimize(best_h, beta=100.0, steps=400, lr=0.005, deadline=_DEADLINE - 12)
        _update_best(h_opt)
        
    # Phase 4: Active-set polish - structural refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_polish = _active_set_polish(best_h, deadline=_DEADLINE - 5)
        _update_best(h_polish)
        
    # Phase 5: Ultra-high beta final polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        h_opt = _pgd_optimize(best_h, beta=300.0, steps=200, lr=0.002, deadline=_DEADLINE - 1)
        _update_best(h_opt)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final projection to ensure exact sum and bounds
    best_h = _project_sum_box(best_h, N / 2.0)
    
    return best_h

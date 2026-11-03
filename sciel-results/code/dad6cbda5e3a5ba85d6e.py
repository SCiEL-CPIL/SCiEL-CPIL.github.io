"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-based smooth-max optimization with Projected Gradient Descent
and robust projection onto the feasible set (sum constraint + box constraints).
Strategy: Annealing log-sum-exp beta with PGD from multiple diverse initializations.
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


def _project_to_feasible(y):
    """
    Project vector y onto the set {h | 0 <= h_i <= 1, sum(h) = N/2}.
    Uses bisection on the threshold tau (Lagrange multiplier for sum constraint).
    """
    n = len(y)
    target_sum = n / 2.0
    
    # Initial bounds for tau
    lo = np.min(y) - 2.0
    hi = np.max(y) + 2.0
    
    # Bisection to find tau such that sum(clip(y - tau, 0, 1)) = target_sum
    for _ in range(80):
        mid = (lo + hi) / 2.0
        h = np.clip(y - mid, 0.0, 1.0)
        s = np.sum(h)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
            
    h = np.clip(y - lo, 0.0, 1.0)
    
    # Correct any numerical drift in the sum
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        # Adjust only elements strictly inside (0, 1) to avoid violating bounds
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    return h


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (log-sum-exp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    
    # Numerical stability
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    # Vectorized gradient accumulation
    # We need to compute sum_k w[k] * dC_k/dh_j
    # dC_k/dh_j is non-zero only for specific j relative to k.
    # Let's use the loop for clarity as N=100 is small, but we can optimize if needed.
    # For N=100, 100 iterations is trivial.
    
    for k in range(n):
        if w[k] < 1e-15:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _pgd_optimize(h0, betas, max_steps_per_beta, base_eta):
    """
    Run Projected Gradient Descent with annealing beta.
    """
    h = h0.copy()
    best_h = h.copy()
    best_score = _score(best_h)
    
    for beta in betas:
        if time.time() > _DEADLINE - 5:
            break
            
        eta = base_eta
        no_improve_count = 0
        prev_obj = float('inf')
        
        for step in range(max_steps_per_beta):
            if time.time() > _DEADLINE - 5:
                break
                
            obj, grad = _smooth_max_grad(h, beta)
            
            # Adaptive step size: decrease if objective increases
            h_new = h - eta * grad
            h_proj = _project_to_feasible(h_new)
            
            # Evaluate true score (max overlap) to track improvement
            s = _score(h_proj)
            
            if s < best_score:
                best_score = s
                best_h = h_proj.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Check for convergence/stagnation
            if abs(prev_obj - obj) < 1e-9 and no_improve_count > 10:
                break
                
            prev_obj = obj
            h = h_proj
            
            # Decay step size slightly
            eta *= 0.998
            
            if no_improve_count > 50:
                break
                
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_to_feasible(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 1. Generate Diverse Initializations
    i = np.arange(N)
    initializations = []
    
    # Cosine wave (fundamental frequency)
    h_cos1 = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    initializations.append(h_cos1)
    
    # Sine wave
    h_sin1 = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    initializations.append(h_sin1)
    
    # Higher frequency cosine
    h_cos2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    initializations.append(h_cos2)
    
    # Alternating pattern
    h_alt = np.where(i % 2 == 0, 0.7, 0.3)
    initializations.append(h_alt)
    
    # Block pattern
    h_block = np.zeros(N)
    h_block[:N//4] = 0.8
    h_block[N//4:3*N//4] = 0.4
    h_block[3*N//4:] = 0.8
    initializations.append(h_block)
    
    # Random
    h_rand = np.random.uniform(0, 1, N)
    initializations.append(h_rand)
    
    # Uniform
    h_unif = np.full(N, 0.5)
    initializations.append(h_unif)
    
    # 2. Optimization with Annealing
    # Beta sequence: start smooth, end sharp
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    max_steps = 200
    base_eta = 0.05
    
    for h0 in initializations:
        if time.time() > _DEADLINE - 10:
            break
            
        h0_proj = _project_to_feasible(h0)
        
        # Run PGD from this start
        h_opt, score = _pgd_optimize(h0_proj, betas, max_steps, base_eta)
        
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
            
    # 3. Final High-Precision Polish
    # If we have a good candidate, try to refine it further with very high beta
    if best_h is not None and time.time() < _DEADLINE - 5:
        # Use a single high beta and many steps
        betas_final = [500.0, 1000.0]
        h_polished, score_polished = _pgd_optimize(best_h, betas_final, 500, 0.01)
        
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final projection to ensure constraints are met exactly
    best_h = _project_to_feasible(best_h)
    
    return best_h

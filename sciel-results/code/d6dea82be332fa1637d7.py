"""Rewritten program: Erdos minimum overlap via bisection on t with Nesterov accelerated projected gradient on smooth-max surrogate."""
import numpy as np
import time

N = 100
DEADLINE = 115.0  # safety margin before 120s kill

def _shifted_overlap(h, k):
    """Compute overlap of h with shifted (1-h) at shift k.
    Overlap = sum_i h[i] * (1 - h[(i-k) % N])
    """
    shifted_h = np.roll(h, k)
    return np.sum(h * (1.0 - shifted_h))

def _max_overlap(h):
    """Maximum overlap over all shifts k=1..N-1 (k=0 is trivially 0 since h*(1-h) <= 0.5 but we exclude self)."""
    # Actually k=0: sum h[i]*(1-h[i]) which is not "overlap with shifted" in the usual sense.
    # The problem says "shifted 1-h", so shifts k=1..N-1.
    max_ov = 0.0
    for k in range(1, N):
        shifted_h = np.roll(h, k)
        ov = np.sum(h * (1.0 - shifted_h))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _smooth_max_overlap(h, beta):
    """Smooth approximation of max overlap using log-sum-exp."""
    overlaps = np.array([_shifted_overlap(h, k) for k in range(1, N)])
    # Log-sum-exp for stability
    m = np.max(overlaps)
    if m < -1e10:
        return 0.0
    log_sum_exp = m + np.log(np.sum(np.exp(beta * (overlaps - m))))
    return log_sum_exp / beta

def _grad_smooth_max_overlap(h, beta):
    """Gradient of smooth max overlap w.r.t. h."""
    overlaps = np.array([_shifted_overlap(h, k) for k in range(1, N)])
    m = np.max(overlaps)
    if m < -1e10:
        return np.zeros(N)
    weights = np.exp(beta * (overlaps - m))
    weights /= np.sum(weights)
    
    grad = np.zeros(N)
    for idx, k in enumerate(range(1, N)):
        w = weights[idx]
        shifted_h = np.roll(h, k)
        # d/dh[i] of sum_j h[j]*(1-h[(j-k)%N])
        # = (1 - h[(i-k)%N]) - h[(i+k)%N]
        # Let's verify: term i contributes h[i]*(1-h[(i-k)%N]), derivative = 1-h[(i-k)%N]
        # Also h[i] appears in term j where (j-k)%N = i, i.e., j = (i+k)%N:
        #   h[(i+k)%N]*(1-h[i]), derivative w.r.t h[i] = -h[(i+k)%N]
        grad[i] += w * (1.0 - shifted_h[i]) - w * h[(i + k) % N]
    
    return grad

def _project(h):
    """Project h onto {h in [0,1]^N : sum h = N/2}."""
    # First clip to [0,1]
    h_clipped = np.clip(h, 0.0, 1.0)
    # Then adjust sum to be N/2
    s = np.sum(h_clipped)
    target = N / 2.0
    if s == 0:
        return np.full(N, 0.5)
    # Simple adjustment: shift all values, then re-clip
    diff = target - s
    h_adj = h_clipped + diff / N
    # If clipping changes sum again, do a few iterations
    for _ in range(50):
        h_clipped = np.clip(h_adj, 0.0, 1.0)
        s = np.sum(h_clipped)
        if abs(s - target) < 1e-12:
            break
        diff = target - s
        h_adj = h_clipped + diff / N
    return h_adj

def _nesterov_smooth(h0, beta, max_iters=2000, tol=1e-8):
    """Nesterov accelerated projected gradient on smooth max overlap."""
    h = h0.copy()
    y = h0.copy()
    t_k = 1.0
    step = 1.0 / beta  # Lipschitz constant is roughly beta (since gradient of LSE has bounded norm)
    # Actually, let's use a smaller step for stability
    step = 0.5 / beta
    
    prev_val = _smooth_max_overlap(y, beta)
    
    for it in range(max_iters):
        # Gradient at y
        grad_y = _grad_smooth_max_overlap(y, beta)
        
        # Nesterov update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
        alpha = (t_k - 1.0) / t_new
        
        y_new = (1.0 + alpha) * h - alpha * y
        y_new = y_new - step * grad_y
        y_new = _project(y_new)
        
        val_new = _smooth_max_overlap(y_new, beta)
        
        # If value increased, fall back
        if val_new > prev_val + 1e-10:
            y_new = h
            val_new = prev_val
        
        h = y_new
        y = y_new
        t_k = t_new
        prev_val = val_new
        
        # Convergence check
        if it > 10 and abs(val_new - _smooth_max_overlap(h, beta)) < tol:
            break
    
    return h, _smooth_max_overlap(h, beta)

def construct():
    start_time = time.time()
    
    # Start with uniform 0.5
    h_best = np.full(N, 0.5)
    val_best = _max_overlap(h_best)
    
    # Try diverse initial patterns
    starts = [np.full(N, 0.5)]
    
    # Multi-frequency patterns
    for freq in [1, 2, 3, 4, 5]:
        t = np.arange(N)
        pattern = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t / N)
        pattern = _project(pattern)
        starts.append(pattern)
    
    # Random starts
    for _ in range(5):
        rand_h = np.random.uniform(0, 1, N)
        rand_h = _project(rand_h)
        starts.append(rand_h)
    
    # Bisection on t (upper bound on max overlap)
    # We want to find the minimum t such that there exists h with max_overlap(h) <= t
    # Use smooth surrogate with increasing beta
    
    lo = 0.0
    hi = _max_overlap(np.full(N, 0.5))  # baseline
    
    # Actually, let's just optimize directly with increasing beta
    betas = [10, 20, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        best_h_in_beta = None
        best_val_in_beta = float('inf')
        
        for h0 in starts:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            h_opt, val_opt = _nesterov_smooth(h0, beta, max_iters=1000)
            # Evaluate true max overlap
            true_val = _max_overlap(h_opt)
            if true_val < best_val_in_beta:
                best_val_in_beta = true_val
                best_h_in_beta = h_opt.copy()
        
        if best_h_in_beta is not None and best_val_in_beta < val_best:
            val_best = best_val_in_beta
            h_best = best_h_in_beta.copy()
        
        # Update starts for next beta iteration
        if best_h_in_beta is not None:
            starts = [best_h_in_beta.copy()] + starts[:3]
    
    # Final polish: try a few more iterations with high beta
    if time.time() - start_time < DEADLINE - 5:
        beta_final = 1000
        h_opt, val_opt = _nesterov_smooth(h_best, beta_final, max_iters=2000)
        true_val = _max_overlap(h_opt)
        if true_val < val_best:
            val_best = true_val
            h_best = h_opt.copy()
    
    # Ensure sum is exactly N/2
    h_best = _project(h_best)
    
    return h_best

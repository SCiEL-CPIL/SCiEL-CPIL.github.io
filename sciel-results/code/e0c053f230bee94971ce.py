"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Full-space Projected Gradient Descent (PGD) with Softmax Objective 
and Coordinate Descent Polish.
Instead of restricting to a low-dimensional parametric manifold (which might 
exclude the global optimum if it requires irregular features), we optimize the 
full 100-dimensional vector directly. We use a smooth approximation of the 
max function (softmax with high temperature) to create a differentiable 
objective. We initialize with a structured "thickened" pure tone to provide 
a good starting point away from the boundaries where gradient flow can be 
stagnant. Finally, we apply Cyclic Coordinate Descent to fine-tune individual 
elements, which is effective for piecewise-linear-like landscapes created by 
the box constraints and the max function.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = 1 - (2/N) * r_k, where r_k is the circular autocorrelation.
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

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    Uses a robust iterative clipping and redistribution.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    for _ in range(50):
        diff = target - np.sum(h)
        if abs(diff) < 1e-15:
            break
        # Identify elements that can accept the correction
        # If diff > 0, we need to increase some elements. Only those < 1 can be increased.
        # If diff < 0, we need to decrease some elements. Only those > 0 can be decreased.
        
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
            
        if not np.any(mask):
            # If no element can move, we are stuck. 
            # This implies sum(h) is already at the boundary of the feasible region 
            # for the direction of correction, which shouldn't happen if we start feasible 
            # and use projection correctly, but as a safety:
            # If we are slightly off, just distribute to the element furthest from the boundary
            if diff > 0:
                idx = np.argmin(h)
            else:
                idx = np.argmax(h)
            h[idx] += diff
            break
            
        # Distribute the correction evenly among the movable elements
        h[mask] += diff / np.sum(mask)
        
        # Clip to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        
    # Final exact fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _softmax_obj_grad(h, beta):
    """
    Compute the softmax approximation of the max overlap and its gradient.
    """
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Numerical stability: subtract max
    exp_cs = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros_like(h)
    n = len(h)
    
    # Gradient of C_k w.r.t h
    # C_k = 1 - (2/n) * sum_{j=0}^{n-1-k} h[j] h[j+k]  (for k>0, careful with indices)
    # Actually, r_k = sum_{j=0}^{n-1-k} h[j] h[j+k]
    # dC_k/dh_j = -(2/n) * ( h[j-k] if j>=k else 0 + h[j+k] if j+k<n else 0 )
    
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            # h[j-k] for j >= k
            # grad[j] gets factor * h[j-k]
            # This is equivalent to: grad[k:] += factor * h[:n-k]
            grad[k:] += factor * h[:n-k]
        
        # h[j+k] for j < n-k
        # grad[j] gets factor * h[j+k]
        # This is equivalent to: grad[:n-k] += factor * h[k:]
        grad[:n-k] += factor * h[k:]
        
    return val, grad

def _pgd_step(h, beta, step_size):
    """
    One step of Projected Gradient Descent.
    """
    val, grad = _softmax_obj_grad(h, beta)
    
    # Gradient step
    h_new = h - step_size * grad
    
    # Projection onto {x: sum(x)=N/2, 0<=x<=1}
    # This is a convex optimization problem: min ||x - h_new||^2 s.t. sum(x)=N/2, 0<=x<=1
    # Can be solved via bisection on the Lagrange multiplier for the sum constraint.
    # x_i = clip(h_new_i - lambda, 0, 1)
    # We need sum(clip(h_new_i - lambda, 0, 1)) = N/2
    
    def sum_clipped(lam):
        return np.sum(np.clip(h_new - lam, 0.0, 1.0))
    
    # Bisection for lambda
    # Range of lambda: min(h_new) - 1 to max(h_new) + 1
    lo = np.min(h_new) - 1.0
    hi = np.max(h_new) + 1.0
    
    target_sum = N / 2.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s = sum_clipped(mid)
        if s > target_sum:
            # Sum is too high, need to subtract more -> increase lambda
            lo = mid
        else:
            # Sum is too low, need to subtract less -> decrease lambda
            hi = mid
            
    lam = (lo + hi) / 2.0
    h_proj = np.clip(h_new - lam, 0.0, 1.0)
    
    # Ensure exact sum in floating point
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-15:
        # Adjust the element furthest from boundary
        if diff > 0:
            idx = np.argmin(h_proj)
        else:
            idx = np.argmax(h_proj)
        h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
        
    return h_proj, val

def _coordinate_descent_polish(h, max_iters=500):
    """
    Cyclic Coordinate Descent to polish the solution.
    For each coordinate i, find the optimal value in [0,1] that minimizes the max overlap,
    while keeping other coordinates fixed and adjusting for the sum constraint?
    Actually, adjusting one coordinate breaks the sum constraint.
    Standard CD for this problem is tricky with the equality constraint.
    
    Alternative: 
    1. Pick a pair of coordinates (i, j).
    2. Vary h_i and h_j such that h_i + h_j = constant (to preserve sum).
    3. Minimize the objective along this line segment.
    
    This is more complex. Let's stick to a simpler "Local Search" that respects the sum.
    Pick i, j. Let delta be the change. h_i += delta, h_j -= delta.
    Bounds: delta in [max(-h_i, h_j-1), min(1-h_i, h_j)].
    Objective is piecewise linear in delta? No, quadratic.
    We can do a ternary search or golden section search on delta.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    n = len(h)
    
    # Pre-compute indices for pairs? Random pairs are better to avoid cycling.
    np.random.seed(42)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
            
        # Pick two random distinct indices
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        h_i = h[i]
        h_j = h[j]
        
        # Range for delta:
        # h_i + delta >= 0 => delta >= -h_i
        # h_i + delta <= 1 => delta <= 1 - h_i
        # h_j - delta >= 0 => delta <= h_j
        # h_j - delta <= 1 => delta >= h_j - 1
        
        lo = max(-h_i, h_j - 1.0)
        hi = min(1.0 - h_i, h_j)
        
        if hi - lo < 1e-12:
            continue
            
        # Golden section search for optimal delta in [lo, hi]
        # Objective: max( C_k(h) ) where h is modified at i, j.
        # We can compute C_k quickly if we update only the affected terms?
        # C_k depends on h[j]h[j+k] and h[j-k]h[j].
        # Changing h_i and h_j affects C_k for k such that i or j is involved.
        # This is complex. Let's just evaluate the full score for simplicity, 
        # but limit the number of evaluations.
        
        # Since N=100, full score eval is fast (~50us).
        # Let's do ~20 evals per coordinate pair update.
        
        phi = (np.sqrt(5) - 1) / 2 # 0.618
        a = lo
        b = hi
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        
        # Store original h to restore
        h_orig = h.copy()
        
        def eval_delta(delta):
            h[i] = h_i + delta
            h[j] = h_j - delta
            return _score(h)
            
        fc = eval_delta(c)
        fd = eval_delta(d)
        
        for _ in range(20):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - phi * (b - a)
                fc = eval_delta(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + phi * (b - a)
                fd = eval_delta(d)
                
        # Best delta is around (c+d)/2
        mid = (c + d) / 2.0
        h[i] = h_i + mid
        h[j] = h_j - mid
        
        # Ensure exact sum
        diff = N/2.0 - np.sum(h)
        if abs(diff) > 1e-15:
             # This shouldn't happen if we are precise, but just in case
             h[i] += diff
             h = np.clip(h, 0.0, 1.0)
             # Re-adjust j to fix sum if clipping broke it? 
             # Actually, if we clip, sum changes.
             # Let's just use _fix_sum for safety if it went out of bounds
             if h[i] < 0 or h[i] > 1 or h[j] < 0 or h[j] > 1:
                 h = _fix_sum(h)

    return h

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
            return True
        return False

    # Initialization: "Thickened" Pure Tone
    # We want a smooth function with significant variance to activate the box constraints
    # and provide a good gradient signal.
    i = np.arange(N)
    # Frequency 1.5, Amplitude 0.8, Phase 0
    h_init = 0.5 + 0.8 * np.sin(2 * np.pi * 1.5 * i / N)
    h_init = _fix_sum(h_init)
    
    # Try a few different initializations to be safe
    inits = [
        h_init,
        _fix_sum(0.5 + 0.9 * np.cos(2 * np.pi * 2.0 * i / N)),
        _fix_sum(np.random.uniform(0.1, 0.9, N)) # Random start
    ]
    
    for h0 in inits:
        if time.time() > _DEADLINE - 20:
            break
            
        h = h0.copy()
        
        # Phase 1: PGD with adaptive step size
        # Start with a moderate step size, decay it
        step_size = 0.5
        beta = 20.0 # Temperature for softmax
        
        for it in range(2000):
            if time.time() > _DEADLINE - 10:
                break
                
            # Adaptive step size: start high, decay
            current_step = step_size * (0.99 ** it)
            if current_step < 1e-4:
                break
                
            h_new, val = _pgd_step(h, beta, current_step)
            
            # Check improvement
            if val < _score(h) + 1e-9:
                h = h_new
            else:
                # If no improvement, reduce step size faster
                step_size *= 0.9
                if step_size < 1e-4:
                    break
                    
        _update_best(h)
        
        # Phase 2: Coordinate Descent Polish
        if time.time() < _DEADLINE - 5:
            h_polished = _coordinate_descent_polish(h, max_iters=500)
            _update_best(h_polished)
            
    if best_h is None:
        best_h = _fix_sum(np.random.uniform(0, 1, N))
        
    return best_h

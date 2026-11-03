"""Rewritten program for Erdos minimum overlap task using FISTA on soft-min surrogate."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 135.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_min_corr(h):
    """Compute min over shifts of cross-correlation R(s) = (2/n)*sum h[i]*h[i+k]."""
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    # Project onto {x >= 0, sum = target_sum}
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    # Clip to [0, 1]
    w = np.clip(w, 0, 1)
    
    # Adjust to ensure sum = target_sum
    for _ in range(20):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    return w

def _soft_min_obj_grad(h, beta):
    """Compute soft-min objective and gradient using FFT."""
    n = len(h)
    
    # Compute correlations R_k for k=1 to n-1
    # R_k = sum_{i=0}^{n-1-k} h[i] h[i+k]
    # Using np.correlate
    corr_full = np.correlate(h, h, mode='full')
    # R_k is at index n-1+k
    R = np.zeros(n)
    R[1:n] = corr_full[n : n-1+n] # indices n to 2n-2 correspond to k=1 to n-1
    
    corrs = R[1:]
    
    # Soft min with numerical stability
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Weights
    weights = exp_terms / sum_exp
    
    # Gradient using FFT
    # grad_corr_j = sum_{k=1}^{n-1} w_k (h_{j+k} + h_{j-k})
    
    w_padded = np.zeros(n)
    w_padded[1:n] = weights
    
    w_rev_padded = np.zeros(n)
    w_rev_padded[:n-1] = weights[::-1]
    
    H = np.fft.fft(h)
    W = np.fft.fft(w_padded)
    W_rev = np.fft.fft(w_rev_padded)
    
    grad_corr = np.fft.ifft(H * (W + W_rev)).real
    
    grad_obj = - (2.0 / n) * grad_corr
    
    return obj, grad_obj

def _fista_optimize(h0, beta, max_iter=5000, tol=1e-8):
    """FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for the constrained problem.
    
    Minimize f(x) = 1 - soft_min_corr(x)
    subject to 0 <= x <= 1, sum(x) = n/2
    
    We use a penalty method: minimize f(x) + mu/2 * ||P(x) - x||^2 where P is projection,
    but actually FISTA works with proximal operator. Here we use the projection as prox.
    
    Actually, for constrained optimization with FISTA:
    - Objective: f(x) = 1 - soft_min(x)
    - Constraint: x in C = {x : 0 <= x <= 1, sum(x) = n/2}
    - Proximal operator: P_C(x) = project to C
    
    FISTA iteration:
    1. y_k = x_k + (t_{k-1} - 1)/(t_k + 1) * (x_k - x_{k-1})
    2. x_{k+1} = P_C(y_k - (1/L) * grad_f(y_k))
    3. t_{k+1} = 0.5 * (1 + sqrt(1 + 4*t_k^2))
    """
    n = len(h0)
    target_sum = n / 2.0
    
    x = h0.copy()
    x_prev = h0.copy()
    t = 1.0
    
    # Estimate Lipschitz constant L
    # We'll use a conservative estimate and adapt if needed
    L = 100.0
    
    best_x = x.copy()
    best_obj = _soft_min_obj_grad(x, beta)[0]
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        # FISTA momentum
        if it > 0:
            t_prev = t
            t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x + (t_prev - 1.0) / (t + 1.0) * (x - x_prev)
        else:
            y = x.copy()
        
        # Gradient step
        obj_y, grad_y = _soft_min_obj_grad(y, beta)
        
        # Backtracking line search for L
        alpha = 1.0
        while alpha > 1e-4:
            z = y - alpha * grad_y
            z_proj = _project_to_simplex(z, target_sum)
            # Check if this step reduces the objective sufficiently
            obj_z, _ = _soft_min_obj_grad(z_proj, beta)
            if obj_z <= obj_y + np.dot(grad_y, z_proj - y) + (1.0/(2.0*alpha)) * np.sum((z_proj - y)**2):
                break
            alpha *= 0.5
        
        x_new = _project_to_simplex(y - alpha * grad_y, target_sum)
        
        # Check convergence
        if np.sum((x_new - x)**2) < tol * tol:
            x = x_new
            break
        
        x_prev = x
        x = x_new
        
        obj_x, _ = _soft_min_obj_grad(x, beta)
        if obj_x < best_obj:
            best_obj = obj_x
            best_x = x.copy()
    
    return best_x, best_obj

def _coordinate_polish(h, max_iter=200):
    """Coordinate descent polish for the hard min objective."""
    n = len(h)
    target_sum = n / 2.0
    h = h.copy()
    
    best_obj = 1.0 - _compute_min_corr(h)
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        
        improved = False
        for j in range(n):
            if _time_remaining() < 5.0:
                break
            
            # Try perturbing coordinate j
            old_val = h[j]
            
            # Try small perturbations
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h[j] = np.clip(old_val + delta, 0, 1)
                
                # Adjust other coordinates to maintain sum
                diff = target_sum - np.sum(h)
                if abs(diff) > 1e-10:
                    mask = np.ones(n, dtype=bool)
                    mask[j] = False
                    cnt = np.sum(mask)
                    if cnt > 0:
                        h[mask] += diff / cnt
                        h = np.clip(h, 0, 1)
                        # Re-adjust
                        diff2 = target_sum - np.sum(h)
                        if abs(diff2) > 1e-10:
                            mask2 = (h > 0) & (h < 1)
                            cnt2 = np.sum(mask2)
                            if cnt2 > 0:
                                h[mask2] += diff2 / cnt2
                                h = np.clip(h, 0, 1)
                
                obj = 1.0 - _compute_min_corr(h)
                if obj < best_obj - 1e-10:
                    best_obj = obj
                    improved = True
                    break
            
            if not improved:
                h[j] = old_val
        
        if not improved:
            break
    
    return h, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_min_corr(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
        restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # More alternating patterns
    for period in [3, 4, 5]:
        pattern = [0.0 if (i % period) < period//2 else 1.0 for i in range(n)]
        mean_val = np.mean(pattern)
        if mean_val > 0:
            pattern = [p / (2*mean_val) for p in pattern]
        restarts.append(np.array(pattern))
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal with various frequencies
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4]:
        restarts.append(0.5 + 0.3*np.sin(freq*x))
        restarts.append(0.5 + 0.3*np.cos(freq*x))
    # Two-tone patterns
    for ratio in [0.2, 0.3, 0.4]:
        k = int(n * ratio)
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 1.0/(1-ratio))]))
    
    # Optimize each restart with FISTA on soft-min objective
    for x0 in restarts:
        if _time_remaining() < 10.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        try:
            # Use FISTA with moderate beta
            x_opt, obj_opt = _fista_optimize(x, beta=50.0, max_iter=2000, tol=1e-8)
            
            # Evaluate with hard min
            x_opt = _project_to_simplex(x_opt, target_sum)
            obj_hard = 1.0 - _compute_min_corr(x_opt)
            
            if obj_hard < best_obj:
                best_obj = obj_hard
                best_x = x_opt
        except Exception:
            pass
    
    # Polish with higher beta for sharper min approximation
    if _time_remaining() > 15.0:
        for beta in [100.0, 200.0, 500.0]:
            if _time_remaining() < 10.0:
                break
            try:
                x_opt, obj_opt = _fista_optimize(best_x, beta=beta, max_iter=3000, tol=1e-9)
                x_opt = _project_to_simplex(x_opt, target_sum)
                obj_hard = 1.0 - _compute_min_corr(x_opt)
                if obj_hard < best_obj:
                    best_obj = obj_hard
                    best_x = x_opt
            except Exception:
                pass
    
    # Final coordinate descent polish
    if _time_remaining() > 10.0:
        best_x, best_obj = _coordinate_polish(best_x, max_iter=100)
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END

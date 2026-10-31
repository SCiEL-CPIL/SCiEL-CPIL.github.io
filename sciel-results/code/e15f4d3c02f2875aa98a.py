import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def smooth_max(conv, beta=50.0):
    """Compute smooth approximation of max using log-sum-exp."""
    max_c = np.max(conv)
    if max_c < 1e-15:
        return 0.0, 0.0
    shifted = conv - max_c
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    smooth_max = max_c + np.log(sum_exp) / beta
    weights = exp_shifted / sum_exp
    return smooth_max, weights

def gradient_of_smooth_max(h, beta=50.0):
    """Compute gradient of smooth max of autoconvolution w.r.t. h."""
    conv = fftconvolve(h, h, mode='full')
    smooth_max_val, weights = smooth_max(conv, beta)
    
    # Gradient of conv[k] w.r.t. h[i] is 2*h[k-i] if 0 <= k-i < N, else 0
    # Gradient of smooth_max w.r.t. h[i] = (1/beta) * sum_k weights[k] * d(conv[k])/dh[i]
    # = (1/beta) * sum_k weights[k] * 2*h[k-i] (with bounds)
    
    grad = np.zeros(N)
    for i in range(N):
        s = 0.0
        for k in range(2 * N - 1):
            if weights[k] > 1e-15:
                ki = k - i
                if 0 <= ki < N:
                    s += weights[k] * h[ki]
                # Also need h[i-k] for the other term in convolution
                ik = i - k
                if 0 <= ik < N:
                    s += weights[k] * h[ik]
        grad[i] = (2.0 / beta) * s
    return grad, smooth_max_val

def coordinate_descent(h, max_iter=500, beta=50.0, deadline=None):
    """
    Coordinate descent on the probability simplex to maximize ratio.
    Uses smooth max surrogate for the supremum.
    """
    h = h.copy()
    h = np.maximum(h, 0.0)
    total = np.sum(h)
    if total < 1e-15:
        return h
    h = h / total
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try moving mass from i to j
            for j in range(N):
                if i == j:
                    continue
                if h[i] < 1e-15:
                    continue
                
                # Try small transfers
                for eps in [1e-4, 1e-3, 5e-3, 1e-2]:
                    if h[i] < eps:
                        continue
                    h_trial = h.copy()
                    h_trial[i] -= eps
                    h_trial[j] += eps
                    h_trial = np.maximum(h_trial, 0.0)
                    t = np.sum(h_trial)
                    if t < 1e-15:
                        continue
                    h_trial = h_trial / t
                    
                    sc = compute_ratio(h_trial)
                    sc_curr = compute_ratio(h)
                    if sc > sc_curr + 1e-12:
                        h = h_trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return h

def parametric_init(kind, n_points=None):
    """Generate initial height vector from parametric families."""
    if n_points is None:
        n_points = N
    
    if kind == 'uniform':
        return np.ones(N) / N
    elif kind == 'ramp':
        h = np.linspace(1.0, N, N)
        return h / np.sum(h)
    elif kind == 'ramp_rev':
        h = np.linspace(N, 1.0, N)
        return h / np.sum(h)
    elif kind == 'triangular':
        h = np.abs(np.linspace(-1, 1, N))
        h[0] = 0.0
        h[-1] = 0.0
        return h / np.sum(h)
    elif kind == 'gaussian':
        x = np.linspace(-1, 1, N)
        h = np.exp(-2.0 * x**2)
        return h / np.sum(h)
    elif kind == 'gaussian_wide':
        x = np.linspace(-1, 1, N)
        h = np.exp(-0.5 * x**2)
        return h / np.sum(h)
    elif kind == 'cosine':
        x = np.linspace(-np.pi/2, np.pi/2, N)
        h = np.cos(x) ** 2
        return h / np.sum(h)
    elif kind == 'sparse_ramp':
        # Sparse version of ramp
        indices = np.arange(0, N, 2)
        h = np.zeros(N)
        for idx in indices:
            h[idx] = 1.0 + 0.1 * idx
        return h / np.sum(h)
    elif kind == 'sparse_cosine':
        indices = np.arange(0, N, 3)
        h = np.zeros(N)
        for idx in indices:
            x = (idx / (N - 1)) * 2 - 1
            h[idx] = np.cos(x * np.pi / 2) ** 2
        return h / np.sum(h)
    elif kind == 'alternating':
        h = np.zeros(N)
        for i in range(0, N, 2):
            h[i] = 1.0
        return h / np.sum(h)
    elif kind == 'clustered':
        # Two clusters
        h = np.zeros(N)
        h[5:15] = 1.0
        h[35:45] = 1.0
        return h / np.sum(h)
    else:
        return np.ones(N) / N

def soft_threshold_polish(h, max_iter=200, beta=50.0, deadline=None):
    """
    Polish using gradient information with soft thresholding.
    """
    h = h.copy()
    h = np.maximum(h, 0.0)
    total = np.sum(h)
    if total < 1e-15:
        return h
    h = h / total
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad, smooth_max_val = gradient_of_smooth_max(h, beta)
        
        # Gradient of ratio = h2_sq / smooth_max
        # d(ratio)/dh_i = (2*h_i * smooth_max - h2_sq * grad_i) / smooth_max^2
        h2_sq = np.sum(h**2)
        ratio_grad = (2.0 * h * smooth_max_val - h2_sq * grad) / (smooth_max_val ** 2)
        
        # Projected gradient step on simplex
        lr = 0.01
        h_trial = h + lr * ratio_grad
        h_trial = np.maximum(h_trial, 0.0)
        t = np.sum(h_trial)
        if t < 1e-15:
            break
        h_trial = h_trial / t
        
        sc_new = compute_ratio(h_trial)
        sc_old = compute_ratio(h)
        
        if sc_new > sc_old:
            h = h_trial
        else:
            # Try smaller step
            lr = 0.001
            h_trial = h + lr * ratio_grad
            h_trial = np.maximum(h_trial, 0.0)
            t = np.sum(h_trial)
            if t < 1e-15:
                break
            h_trial = h_trial / t
            sc_new = compute_ratio(h_trial)
            if sc_new > sc_old:
                h = h_trial
            else:
                break
    
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = compute_ratio(best_h)
    
    # List of parametric initializations
    init_kinds = [
        'uniform', 'ramp', 'ramp_rev', 'triangular',
        'gaussian', 'gaussian_wide', 'cosine',
        'sparse_ramp', 'sparse_cosine', 'alternating', 'clustered'
    ]
    
    # Multi-start optimization
    for kind in init_kinds:
        if time.time() > deadline - 20.0:
            break
        
        h_init = parametric_init(kind)
        
        # Try different beta values for smooth max
        for beta in [30.0, 50.0, 100.0]:
            if time.time() > deadline - 15.0:
                break
            
            # Soft threshold polish
            h_polished = soft_threshold_polish(h_init, max_iter=150, beta=beta, deadline=deadline - 10.0)
            sc = compute_ratio(h_polished)
            if sc > best_score:
                best_score = sc
                best_h = h_polished.copy()
            
            # Coordinate descent on the polished result
            if time.time() > deadline - 10.0:
                break
            h_cd = coordinate_descent(h_polished, max_iter=100, beta=beta, deadline=deadline - 5.0)
            sc = compute_ratio(h_cd)
            if sc > best_score:
                best_score = sc
                best_h = h_cd.copy()
        
        # Also try coordinate descent directly on the parametric init
        if time.time() > deadline - 5.0:
            break
        h_cd_direct = coordinate_descent(h_init, max_iter=80, beta=50.0, deadline=deadline - 2.0)
        sc = compute_ratio(h_cd_direct)
        if sc > best_score:
            best_score = sc
            best_h = h_cd_direct.copy()
    
    # Final polish on the best result
    if time.time() > deadline - 3.0:
        return list(best_h)
    
    h_final = soft_threshold_polish(best_h, max_iter=100, beta=50.0, deadline=deadline - 1.0)
    sc_final = compute_ratio(h_final)
    if sc_final > best_score:
        best_h = h_final
    
    return list(best_h)

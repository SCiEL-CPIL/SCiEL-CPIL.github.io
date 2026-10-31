"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _local_sign_flip(h, max_passes=5, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _gradient_descent(h, deadline):
    """
    Gradient descent on the smooth objective with soft-max penalty.
    Objective: minimize max_k |ac(k)| subject to sum(h) = N.
    Using Lagrangian: L = max_k |ac(k)| - lambda * (sum(h) - N)
    Or using soft-max: S = logsumexp(beta * |ac|) / beta
    """
    h = h.copy()
    # Normalize to sum = N
    s = np.sum(h)
    if abs(s) > 1e-9:
        h = h * (N / s)
    best_val = _score(h)
    
    size = 2 * N - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    beta = 1.0  # Softness parameter for logsumexp
    lr = 0.05
    
    for step in range(300):
        if time.time() > deadline:
            break
        
        # Compute ac
        H = np.fft.rfft(h, fsize)
        ac = np.fft.irfft(H * H, fsize)[:size]
        
        # Soft-max of |ac|
        abs_ac = np.abs(ac)
        # Use logsumexp for numerical stability
        max_ac = np.max(abs_ac)
        if max_ac > 0:
            log_sum_exp = np.log(np.sum(np.exp(abs_ac - max_ac))) + max_ac
            soft_max = log_sum_exp
            # Gradient of soft_max w.r.t h
            # d(soft_max)/d(ac_k) = exp(abs_ac_k - max_ac) / sum(exp(abs_ac - max_ac))
            weights = np.exp(abs_ac - max_ac)
            weights /= np.sum(weights)
            # d(soft_max)/d(ac_k) = weights[k] * sign(ac[k])
            grad_ac = weights * np.sign(ac)
        else:
            soft_max = 0
            grad_ac = np.zeros(size)
        
        # Gradient of soft_max w.r.t h
        # soft_max = f(ac(h)), where ac = ifft(fft(h)^2)
        # We need d(soft_max)/dh_i
        # This is complex. Let's use the subgradient of the hard max instead.
        # Find k* = argmax |ac|
        k_star = np.argmax(abs_ac)
        sign_ac = np.sign(ac[k_star])
        
        grad_M = np.zeros(N)
        if k_star == 0:
            grad_M = 2.0 * h
        else:
            idx1 = np.arange(N - k_star)
            grad_M[idx1] += sign_ac * h[idx1 + k_star]
            idx2 = np.arange(k_star, N)
            grad_M[idx2] += sign_ac * h[idx2 - k_star]
        
        # Lagrangian gradient: grad_M - lambda * 1
        # We want to minimize max|ac| subject to sum(h) = N
        # Use projected gradient: h_new = h - lr * grad_M, then normalize to sum = N
        
        h_new = h - lr * grad_M
        
        # Normalize to sum = N
        s_new = np.sum(h_new)
        if abs(s_new) > 1e-9:
            h_new = h_new * (N / s_new)
        else:
            h_new = h
        
        val_new = _score(h_new)
        
        if val_new < best_val:
            best_val = val_new
            h = h_new
            # Decay learning rate
            lr *= 0.99
        else:
            lr *= 0.95
            
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Phase 1: Gradient descent from ones(N)
    h_grad, val_grad = _gradient_descent(best_h, deadline)
    if val_grad < best_val:
        best_val = val_grad
        best_h = h_grad
    
    # Phase 2: Local sign flips on the gradient result
    h_opt, val_opt = _local_sign_flip(best_h, max_passes=5, deadline=deadline)
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt
    
    # Phase 3: Random restarts with gradient descent
    n_restarts = 25
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        if r % 2 == 0:
            h = np.random.uniform(-1, 1, size=N)
        else:
            h = np.random.choice([-1.0, 1.0], size=N)
            
        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            h = h + 0.5
            h = np.sign(h)
            
        # Gradient descent
        h_grad, val_grad = _gradient_descent(h, deadline)
        
        # Local sign flips
        h_opt, val_opt = _local_sign_flip(h_grad, max_passes=3, deadline=deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt
            
    return best_h

# EVOLVE-BLOCK-END

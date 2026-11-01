import numpy as np
import time

N = 400

def _compute_metric(h):
    """
    Compute the max absolute sidelobe of the autocorrelation.
    h is a 1D array of length N with values +/- 1.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    sidelobes = np.abs(ac[1:N])
    return np.max(sidelobes)

def _objective_and_gradient(x, beta=10.0):
    """
    Objective: Soft-max of sidelobes (smooth surrogate for max).
    Gradient: Analytic via chain rule.
    x is a continuous vector in [-1, 1].
    """
    h = x
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    
    sidelobes_abs = np.abs(ac[1:N])
    
    max_val = np.max(sidelobes_abs)
    if max_val < 1e-10:
        return max_val, np.zeros(N)
    
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted))
    obj = max_val + log_sum / beta
    
    # Weights for each sidelobe in the soft-max
    weights = exp_shifted / np.sum(exp_shifted)
    
    # Gradient computation
    # d(obj)/d(h_i) = sum_k weights[k] * sign(ac[k]) * d(ac[k])/d(h_i)
    # d(ac[k])/d(h_i) = h[i+k] + h[i-k] (if valid)
    
    grad = np.zeros(N)
    signs = np.sign(ac[1:N])
    
    for k in range(1, N):
        w_k = weights[k-1]
        if w_k < 1e-15:
            continue
        s_k = signs[k-1]
        coeff = w_k * s_k
        
        # Contribution to indices i where i+k < N: h[i+k]
        # Contribution to indices i where i-k >= 0: h[i-k]
        grad[0:N-k] += coeff * h[k:N]
        grad[k:N] += coeff * h[0:N-k]
        
    return obj, grad

def _continuous_optimize(x0, time_limit=10.0, step_size=0.05, beta=10.0):
    """
    Projected gradient descent with restarts and adaptive step size.
    Projects back to [-1, 1] after each step.
    """
    start_time = time.time()
    x = x0.copy()
    current_obj, _ = _objective_and_gradient(x, beta)
    best_x = x.copy()
    best_obj = current_obj
    
    # Adaptive step size
    lr = step_size
    patience = 0
    max_patience = 100
    
    for _ in range(5000): # High cap, controlled by time
        if time.time() - start_time > time_limit:
            break
        
        obj, grad = _objective_and_gradient(x, beta)
        
        # Gradient step
        x_new = x - lr * grad
        
        # Projection to [-1, 1]
        x_new = np.clip(x_new, -1.0, 1.0)
        
        # Check improvement
        obj_new, _ = _objective_and_gradient(x_new, beta)
        
        if obj_new < current_obj:
            x = x_new
            current_obj = obj_new
            patience = 0
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = x.copy()
        else:
            patience += 1
            # Reduce step size if stuck
            if patience > max_patience // 2:
                lr *= 0.9
            # Occasionally restart from a slightly perturbed best solution
            if patience > max_patience:
                x = best_x + 0.1 * np.random.randn(N)
                x = np.clip(x, -1.0, 1.0)
                current_obj, _ = _objective_and_gradient(x, beta)
                patience = 0
                lr = step_size
    
    return best_x, best_obj

def _local_search(h, time_limit=5.0):
    """
    Local search via random bit flips.
    Returns the best sequence found.
    """
    start_time = time.time()
    current_h = h.copy()
    current_val = _compute_metric(current_h)
    best_h = current_h.copy()
    best_val = current_val
    
    n_fft = 2 * N
    
    for _ in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        # Find the lag with the maximum sidelobe
        F = np.fft.rfft(current_h, n=n_fft)
        power = F * np.conj(F)
        ac = np.fft.irfft(power, n=n_fft)
        sidelobes = np.abs(ac[1:N])
        max_lag = np.argmax(sidelobes) + 1
        
        # Try flipping a bit that contributes significantly to ac[max_lag]
        # We pick a random bit for simplicity, but could be smarter.
        j = np.random.randint(N)
        old_h_j = current_h[j]
        current_h[j] = -old_h_j
        
        # Recompute metric
        F = np.fft.rfft(current_h, n=n_fft)
        power = F * np.conj(F)
        ac_new = np.fft.irfft(power, n=n_fft)
        new_val = np.max(np.abs(ac_new[1:N]))
        
        if new_val < current_val:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = current_h.copy()
        else:
            # Revert
            current_h[j] = old_h_j
            
    return best_h, best_val

def _generate_chirp(N):
    """
    Generate a chirp sequence.
    """
    i = np.arange(N)
    phase = np.pi * i**2 / N
    h = np.sign(np.cos(phase))
    h[h == 0] = 1
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous Optimization
    # Try multiple random initializations in [-1, 1]
    num_starts = 5
    time_per_start = 20.0
    
    for i in range(num_starts):
        if time.time() >= deadline - 20.0:
            break
            
        # Initialize x in [-1, 1]
        # Mix of random and structured starts
        if i == 0:
            x0 = np.ones(N) * 0.9
        elif i == 1:
            x0 = _generate_chirp(N).astype(float) * 0.9
        else:
            x0 = np.random.choice([-1.0, 1.0], size=N) * 0.8
            
        # Run continuous optimization
        x_opt, obj_opt = _continuous_optimize(x0, time_limit=time_per_start)
        
        # Round to binary
        h_bin = np.sign(x_opt)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
            
        # Evaluate binary solution
        val = _compute_metric(h_bin)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
            
    # Phase 2: Discrete Local Search
    if best_h is not None and time.time() < deadline - 5.0:
        time_limit = deadline - 5.0 - time.time()
        best_h, best_val = _local_search(best_h, time_limit=time_limit)
        
    if best_h is None:
        return np.ones(N)
        
    return best_h

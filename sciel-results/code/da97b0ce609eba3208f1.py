import numpy as np
import time

N = 400
TIME_LIMIT = 145  # Safety margin before 150s kill

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    L = 2 * N - 1
    # Next power of 2 for FFT efficiency
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    C = np.fft.irfft(X * X, n=L_fft)[:L]
    return C

def _objective(x):
    """Compute max(|autoconv|) / (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf
    ac = _autoconv_fft(x)
    return np.max(np.abs(ac)) / (s * s)

def _smooth_grad(x, tau):
    """
    Compute gradient of soft-max approximation of max(|ac|)/(sum x)^2.
    Assumes sum(x) is normalized to 1 for simplicity in gradient direction,
    but we handle the scaling in the update step.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    # Normalize x for gradient calculation to simplify
    # The objective is homogeneous, so we can work on the manifold sum(x)=1
    x_norm = x / s
    ac = _autoconv_fft(x_norm)
    abs_ac = np.abs(ac)
    L = len(ac)
    
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    # Soft-max weights
    # Shift to prevent overflow
    shifted = (abs_ac - m) / tau
    shifted = np.clip(shifted, -50, 0)
    weights = np.exp(shifted)
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return m / (s * s), np.zeros_like(x)
    weights /= w_sum
    
    # v_k = weights_k * sign(ac_k)
    v = weights * np.sign(ac)
    
    # Gradient of LSE(|ac|) w.r.t x_norm is 2 * (v * x_norm)
    # We compute the convolution v * x_norm
    L_fft = 1
    while L_fft < L + N:
        L_fft *= 2
        
    v_pad = np.zeros(L_fft)
    v_pad[:L] = v
    
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x_norm
    
    V = np.fft.rfft(v_pad, n=L_fft)
    X = np.fft.rfft(x_pad, n=L_fft)
    
    # Linear convolution
    conv = np.fft.irfft(V * X, n=L_fft)[:N]
    
    grad_norm = 2.0 * conv
    
    # Scale gradient back to original x scale
    # d/dx (LSE(x/s) / s^2) ... 
    # Actually, if we normalize x after each step, we just need the direction.
    # The direction is -grad_norm.
    
    val = (m + tau * np.log(w_sum)) / (s * s)
    
    return val, grad_norm

def construct():
    START_TIME = time.time()
    
    def _time_ok():
        return (time.time() - START_TIME) < TIME_LIMIT
    
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Diverse Initializations
    initializations = [
        np.ones(N),
        np.linspace(2.0, 1.0, N),
        np.linspace(1.0, 2.0, N),
    ]
    # Exponentials
    for r in [0.9, 0.8, 0.7, 0.5, 0.3, 1.2, 1.5, 2.0]:
        initializations.append(r ** np.arange(N))
        
    # Sine/Cosine
    t = np.linspace(0, 1, N)
    for freq in [1, 2, 3, 5, 10]:
        initializations.append(1.0 + 0.5 * np.sin(2 * np.pi * freq * t))
        initializations.append(np.sign(np.sin(2 * np.pi * freq * t)))
        
    # Random
    for _ in range(20):
        initializations.append(rng.normal(1.0, 0.5, size=N))
    
    # Sign patterns
    for period in [2, 4, 8, 16]:
        pattern = np.ones(N)
        pattern[::2] = -1.0
        initializations.append(pattern)
        
    # Update best from initializations
    for x0 in initializations:
        if not _time_ok():
            break
        # Ensure sum is not zero
        s0 = np.sum(x0)
        if abs(s0) < 1e-10:
            continue
        x0 = x0 / s0 # Normalize
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
            
    # Phase 2: Optimization Loop
    # We will run multiple restarts from the best found so far or new random points
    num_restarts = 10
    tau_start = 0.1
    tau_end = 0.001
    lr_start = 0.05
    
    for restart in range(num_restarts):
        if not _time_ok():
            break
            
        # Start from current best or a new random point
        if restart == 0:
            x = best_x.copy()
        else:
            # Perturb best or new random
            if rng.random() < 0.5:
                x = best_x + rng.normal(0, 0.1, size=N)
            else:
                x = rng.normal(1.0, 0.5, size=N)
                
        # Normalize
        s = np.sum(x)
        if abs(s) < 1e-10:
            continue
        x = x / s
        
        current_val = _objective(x)
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
            
        tau = tau_start
        lr = lr_start
        
        for gd_iter in range(20000):
            if not _time_ok():
                break
                
            val_s, grad = _smooth_grad(x, tau)
            if not np.isfinite(val_s):
                break
                
            # Project gradient onto subspace orthogonal to ones? 
            # Since we normalize, the direction -grad is fine, but we must renormalize.
            
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-10:
                break
                
            # Step
            # Use a simple line search or fixed step with decay
            x_trial = x - lr * grad
            
            # Renormalize
            s_trial = np.sum(x_trial)
            if abs(s_trial) < 1e-10:
                break
            x_trial = x_trial / s_trial
            
            trial_val = _objective(x_trial)
            
            if trial_val < current_val:
                x = x_trial
                current_val = trial_val
                if current_val < best_val:
                    best_val = current_val
                    best_x = x.copy()
                # Increase lr slightly if improving
                lr = min(0.2, lr * 1.05)
            else:
                # Decrease lr
                lr *= 0.7
                if lr < 1e-6:
                    break
                    
            # Decay tau
            tau = max(tau_end, tau * 0.995)
            
            # Periodic check
            if gd_iter % 100 == 0:
                pass

    # Phase 3: Final Polish with very small tau
    if _time_ok():
        x = best_x.copy()
        s = np.sum(x)
        x = x / s
        current_val = _objective(x)
        tau = 0.001
        lr = 0.01
        for gd_iter in range(10000):
            if not _time_ok():
                break
            val_s, grad = _smooth_grad(x, tau)
            if not np.isfinite(val_s):
                break
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-12:
                break
            x_trial = x - lr * grad
            s_trial = np.sum(x_trial)
            if abs(s_trial) < 1e-10:
                break
            x_trial = x_trial / s_trial
            trial_val = _objective(x_trial)
            if trial_val < current_val:
                x = x_trial
                current_val = trial_val
                if current_val < best_val:
                    best_val = current_val
                    best_x = x.copy()
            else:
                lr *= 0.9
                if lr < 1e-8:
                    break

    return best_x

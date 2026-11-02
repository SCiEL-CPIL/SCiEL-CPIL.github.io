import numpy as np
import time

N = 400
L = 2 * N - 1

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT."""
    # Use next power of 2 for FFT efficiency
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

def _smooth_obj_grad(x, tau=0.1):
    """
    Compute the log-sum-exp approximation of max(|ac|) / (sum x)^2 
    and its gradient.
    
    Returns:
    val: the smooth objective value
    grad: the gradient w.r.t. x
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    
    # Soft-max approximation of max(abs_ac)
    # L_tau = tau * log(sum(exp(abs_ac / tau)))
    # To avoid overflow, subtract max
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
        
    shifted = (abs_ac - m) / tau
    shifted = np.clip(shifted, -50, 0) # Numerical stability
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-15:
        return m / (s * s), np.zeros_like(x)
        
    # Weights for the soft-max
    weights = exp_shifted / sum_exp
    
    # Soft max value
    soft_max = m + tau * np.log(sum_exp)
    
    # Objective value
    val = soft_max / (s * s)
    
    # Gradient calculation
    # d/dx [ soft_max / s^2 ] = (grad_soft_max * s^2 - soft_max * 2 * s * grad_s) / s^4
    # grad_s = 1 (since s = sum x)
    # So, grad_obj = grad_soft_max / s^2 - 2 * soft_max / s^3
    
    # grad_soft_max = sum_k weights_k * d/dx |ac_k|
    # d/dx |ac_k| = sign(ac_k) * d/dx ac_k
    # d/dx ac_k = convolve(x, e_k) + correlate(x, e_k) ... essentially
    # Let v_k = weights_k * sign(ac_k)
    # Then grad_soft_max_j = sum_k v_k * (x_{k-j} + x_{j-k})
    
    v = weights * np.sign(ac)
    
    # Compute gradient using FFT
    # Term 1: sum_k v_k x_{k-j} (Cross-correlation of v and x)
    # Term 2: sum_k v_k x_{j-k} (Convolution of v and x)
    
    L_fft = 1
    while L_fft < L + N:
        L_fft *= 2
        
    v_pad = np.zeros(L_fft)
    v_pad[:L] = v
    
    x_pad = np.zeros(L_fft)
    x_pad[:N] = x
    
    V = np.fft.rfft(v_pad, n=L_fft)
    X = np.fft.rfft(x_pad, n=L_fft)
    
    # corr1[j] = sum_k v_k x_{k-j}
    # This is cross-correlation. In FFT: IFFT(V * conj(X))
    corr1 = np.fft.irfft(V * np.conj(X), n=L_fft)
    
    # conv2[j] = sum_k v_k x_{j-k}
    # This is convolution. In FFT: IFFT(V * X)
    conv2 = np.fft.irfft(V * X, n=L_fft)
    
    grad_soft_max = corr1[:N] + conv2[:N]
    
    grad_obj = grad_soft_max / (s * s) - 2.0 * soft_max / (s * s * s)
    
    return val, grad_obj

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 145  # Safety margin for 150s limit
    
    def _time_ok():
        return (time.time() - START_TIME) < TIME_LIMIT
    
    best_x = np.ones(N)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. All ones (baseline)
    initializations.append(np.ones(N))
    
    # 2. Random signs
    for _ in range(10):
        initializations.append(rng.choice([-1, 1], size=N).astype(float))
        
    # 3. Structured patterns
    t = np.linspace(0, 1, N)
    for freq in [1, 2, 3, 5, 10]:
        initializations.append(np.sign(np.sin(2 * np.pi * freq * t)))
        initializations.append(1.0 + 0.5 * np.sign(np.sin(2 * np.pi * freq * t)))
        
    # 4. Gaussian bumps
    for center in [0.2, 0.5, 0.8]:
        for width in [0.1, 0.2, 0.5]:
            bump = np.exp(-((t - center) ** 2) / (2 * width ** 2))
            initializations.append(bump)
            
    # 5. Random continuous
    for _ in range(10):
        initializations.append(rng.normal(0, 1, size=N))
        
    # Evaluate initializations
    for x0 in initializations:
        if not _time_ok():
            break
        # Normalize sum to 1 to avoid scaling issues during optimization?
        # The objective is scale invariant: f(cx) = f(x).
        # So we can fix sum(x) = 1 to reduce dimensionality, 
        # but the gradient descent handles it naturally if we just minimize.
        # However, fixing sum=1 might help convergence.
        # Let's just use the raw values.
        
        val = _objective(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()
            
    # Optimization Loop with Adam
    # We will run multiple restarts from the best found so far and random points
    
    max_restarts = 5
    for r in range(max_restarts):
        if not _time_ok():
            break
            
        # Start from best_x or a random point
        if r == 0:
            x = best_x.copy()
        else:
            # Perturb best_x or pick a random init
            if rng.random() > 0.5:
                x = best_x + rng.normal(0, 0.5, size=N)
            else:
                x = rng.normal(0, 1, size=N)
                
        # Adam Optimizer State
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t_adam = 0
        
        lr = 0.01
        tau = 0.1
        patience = 0
        max_patience = 1000
        
        for step in range(50000):
            if not _time_ok():
                break
                
            # Compute objective and gradient
            val, grad = _smooth_obj_grad(x, tau)
            
            if not np.isfinite(val):
                break
                
            # Update best
            if val < best_val:
                best_val = val
                best_x = x.copy()
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    break
                    
            # Adam Update
            t_adam += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t_adam)
            v_hat = v / (1 - beta2 ** t_adam)
            
            x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Normalize x to prevent drift in scale (optional, but objective is scale invariant)
            # Normalizing to sum=1 can help stability
            s = np.sum(x)
            if abs(s) > 1e-6:
                x = x / s
                
            # Adaptive learning rate and tau
            if step % 100 == 0:
                # Decay tau to sharpen the max approximation
                if tau > 0.01:
                    tau *= 0.95
                # Slightly decay lr if stuck
                if patience > 100:
                    lr *= 0.9
                else:
                    lr = min(0.05, lr * 1.05)
                    
    # Final fine-tuning with exact objective and small steps
    if _time_ok():
        lr_final = 0.001
        for step in range(5000):
            if not _time_ok():
                break
            val_exact = _objective(best_x)
            if val_exact < best_val:
                best_val = val_exact
            else:
                best_val = val_exact # Sync best_val with exact metric
                
            # Use smooth gradient for direction, but step with exact check?
            # Simpler: just do a local random search or small gradient step
            val_smooth, grad = _smooth_obj_grad(best_x, tau=0.01)
            
            # Normalized gradient step
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-10:
                break
                
            direction = -grad / norm_grad
            x_trial = best_x + lr_final * direction
            
            # Ensure sum is not zero
            if abs(np.sum(x_trial)) < 1e-6:
                x_trial += 1e-6
                
            val_trial = _objective(x_trial)
            if val_trial < best_val:
                best_x = x_trial
                best_val = val_trial
            else:
                lr_final *= 0.9
                if lr_final < 1e-6:
                    break

    return best_x

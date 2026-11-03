"""Rewritten program: Subgradient/Smooth optimization for low off-diagonal autocorrelation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    """Linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    c = np.fft.irfft(X * X, fsize)[:size]
    return c

def _metric_offdiag(x):
    """Compute max |c_k| for k != 0 divided by (sum x)^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv(x)
    # Exclude k=0
    if len(c) > 1:
        max_val = np.max(np.abs(c[1:]))
    else:
        max_val = 0.0
    return max_val / (s * s)

def _objective_smooth(x, temp=1.0):
    """Smooth approximation of the off-diagonal metric for optimization."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return 1e10
    c = _autoconv(x)
    if len(c) > 1:
        abs_c = np.abs(c[1:])
    else:
        return 0.0
    # Soft max: temp * log(sum(exp(c/temp)))
    max_c = np.max(abs_c)
    soft_max = max_c + temp * np.log(np.sum(np.exp((abs_c - max_c) / temp)))
    return soft_max / (s * s)

def _gradient_descent_polish(x0, max_iter=500, initial_lr=0.01, deadline=None):
    """Projected gradient descent with annealed temperature for final polish."""
    x = x0.copy().astype(float)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return x
    
    # Normalize sum to 1
    x = x / s
    
    best_x = x.copy()
    best_val = _metric_offdiag(best_x)
    
    lr = initial_lr
    temp = 0.5
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute autoconvolution
        c = _autoconv(x)
        if len(c) <= 1:
            break
        abs_c = np.abs(c[1:])
        
        # Soft max gradient
        max_c = np.max(abs_c)
        if max_c < 1e-15:
            break
        
        # Compute soft max and its gradient
        exp_vals = np.exp((abs_c - max_c) / temp)
        sum_exp = np.sum(exp_vals)
        soft_max = max_c + temp * np.log(sum_exp)
        
        # Gradient of soft_max w.r.t. abs_c
        grad_abs_c = exp_vals / sum_exp
        
        # Gradient w.r.t. c (chain rule: d|c|/dc = sign(c))
        c_off = c[1:]
        sign_c = np.sign(c_off)
        grad_c = grad_abs_c * sign_c
        
        # Gradient of autoconvolution w.r.t. x
        # c_k = sum_i x_i * x_{i+k}, so dc_k/dx_j = x_{j-k} + x_{j+k} (with bounds)
        # This is expensive to compute directly; use FFT-based approach
        # For simplicity, use numerical gradient or approximate
        
        # Instead, use the fact that the gradient of sum_k w_k * c_k w.r.t. x
        # can be computed via cross-correlation
        # grad_x[j] = sum_k w_k * (x[j-k] + x[j+k]) where w_k = grad_c[k-1]
        
        # Build the weight vector for cross-correlation
        w = np.zeros(2 * N - 1)
        w[1:] = grad_c
        
        # Cross-correlation: grad_x[j] = sum_k w[k] * x[j-k] (with proper indexing)
        # Using FFT: if W = fft(w), then grad_x = ifft(W * conj(X))
        fsize = 1
        while fsize < 2 * N:
            fsize *= 2
        X = np.fft.rfft(x, fsize)
        W = np.fft.rfft(w, fsize)
        grad_x = np.fft.irfft(W * np.conj(X), fsize)[:N]
        
        # Also need to account for the (sum x)^2 denominator
        s = np.sum(x)
        metric = soft_max / (s * s)
        
        # Gradient of metric = (grad_soft_max * s^2 - soft_max * 2*s * ones) / s^4
        # = (grad_soft_max / s^2) - 2 * soft_max * ones / s^3
        grad_metric = grad_x / (s * s) - 2.0 * soft_max * np.ones(N) / (s ** 3)
        
        # Update
        x_new = x - lr * grad_metric
        
        # Project: keep sum non-zero (scale to sum=1 if needed)
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-12:
            x_new = x + 1e-6 * np.random.randn(N)
            s_new = np.sum(x_new)
        x_new = x_new / s_new
        
        # Evaluate
        val = _metric_offdiag(x_new)
        if val < best_val:
            best_val = val
            best_x = x_new.copy()
        
        # Decay learning rate and temperature
        x = x_new
        lr *= 0.995
        temp *= 0.999
        
    return best_x

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    best_x = np.ones(N)
    best_val = _metric_offdiag(best_x)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    
    # 2. Random sequences with non-zero mean
    for _ in range(15):
        r = rng.random(N)
        # Bias towards positive to ensure non-zero sum
        x = np.where(r > 0.35, 1.0, -1.0)
        if abs(np.sum(x)) > 5:
            candidates.append(x)
            
    # 3. M-sequences (if available, otherwise approximate)
    # Simple pseudo-random binary sequence
    for _ in range(10):
        x = rng.choice([-1, 1], N)
        if abs(np.sum(x)) > 5:
            candidates.append(x)

    # 4. Structured: Block patterns
    for block_size in [5, 10, 20, 50, 100]:
        pattern = np.zeros(N)
        for i in range(N):
            if (i // block_size) % 2 == 0:
                pattern[i] = 1.0
            else:
                pattern[i] = -1.0
        # Add small bias to ensure sum != 0
        pattern[0] += 0.5
        candidates.append(pattern)
    
    # 5. Sine wave patterns with random phase
    for freq in [1, 2, 5, 10]:
        for phase in [0, np.pi/2, np.pi]:
            t = np.arange(N)
            pattern = np.sin(2 * np.pi * freq * t / N + phase)
            # Convert to signed step function
            pattern = np.sign(pattern)
            # Ensure non-zero sum
            if abs(np.sum(pattern)) < 1:
                pattern[0] = 1.0
            candidates.append(pattern)

    # Optimize candidates
    for x0 in candidates:
        if time.time() > deadline - 30:
            break
            
        x0 = x0.copy()
        # Normalize sum to 1 to handle scale invariance
        s0 = np.sum(x0)
        if abs(s0) < 1e-6:
            continue
        x0 = x0 / s0
        
        # Use L-BFGS-B with smooth objective, annealing temperature
        try:
            # First pass with higher temperature
            res1 = minimize(
                lambda x: _objective_smooth(x, temp=1.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_mid = res1.x
            
            # Second pass with lower temperature
            res2 = minimize(
                lambda x: _objective_smooth(x, temp=0.1),
                x_mid,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = res2.x
            
            val = _metric_offdiag(x_opt)
            if val < best_val:
                best_val = val
                best_x = x_opt.copy()
        except Exception:
            pass

    # Final polish with gradient descent on the best candidate
    if time.time() < deadline - 5:
        try:
            polished = _gradient_descent_polish(
                best_x, 
                max_iter=1000, 
                initial_lr=0.005,
                deadline=deadline - 5
            )
            val = _metric_offdiag(polished)
            if val < best_val:
                best_val = val
                best_x = polished
        except Exception:
            pass

    return best_x

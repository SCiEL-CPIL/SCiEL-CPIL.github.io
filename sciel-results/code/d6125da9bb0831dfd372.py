import numpy as np
import time

N = 100
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _autoconv(x):
    """Compute full linear autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.rfft(x, n=n)
    C = np.fft.irfft(X * X, n=n).real[:2 * N - 1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(c)) / (s * s)

def _gradient_max_abs_conv(x):
    """
    Compute subgradient of max|c_k| w.r.t x.
    c_k = sum_j x_j x_{j+k}
    d c_k / d x_i = x_{i+k} + x_{i-k} (with 0 outside range)
    """
    c = _autoconv(x)
    abs_c = np.abs(c)
    k_star = np.argmax(abs_c)
    sign_c = np.sign(c[k_star])
    
    # Compute gradient of c_{k_star}
    # grad[i] = x[i + k_star] + x[i - k_star]
    grad = np.zeros(N)
    
    # Term 1: x[i + k_star]
    # i ranges 0..N-1
    # i + k_star ranges k_star .. N-1+k_star
    # Valid indices for x are 0..N-1
    # So i + k_star < N => i < N - k_star
    if k_star < N:
        idx1 = np.arange(0, N - k_star)
        vals1 = x[idx1 + k_star]
        grad[idx1] += vals1
    
    # Term 2: x[i - k_star]
    # i - k_star >= 0 => i >= k_star
    if k_star < N:
        idx2 = np.arange(k_star, N)
        vals2 = x[idx2 - k_star]
        grad[idx2] += vals2
        
    return sign_c * grad

def _optimize_sequence(x0, max_iter, deadline):
    """
    Minimize max|c_k| subject to sum(x)=1.
    """
    x = x0.copy()
    # Normalize to sum 1
    s = np.sum(x)
    if abs(s) < 1e-10:
        x += 1e-10
        s = np.sum(x)
    x = x / s
    
    best_val = _eval_ratio(x)
    best_x = x.copy()
    
    eta = 0.1
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        grad = _gradient_max_abs_conv(x)
        
        # Gradient step
        x = x - eta * grad
        
        # Project to sum(x) = 1
        s = np.sum(x)
        x = x - (s - 1.0) / N
        
        # Evaluate
        val = _eval_ratio(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
        # Adaptive step size: if stuck, reduce eta
        if it % 100 == 0 and it > 0:
             # Simple heuristic: check improvement
             pass # Keeping it simple for speed

    return best_x, best_val

def construct():
    """
    Return signed step heights that minimize max|c_k| / (sum x_i)^2.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    # Start with constant sequence
    x_const = np.ones(N)
    best_val = _eval_ratio(x_const)
    best_result = x_const.copy()
    
    np.random.seed(42)
    
    # Try several random initializations
    for trial in range(50):
        if time.time() > deadline:
            break
            
        # Random start
        x0 = np.random.randn(N)
        
        x_opt, val = _optimize_sequence(x0, max_iter=500, deadline=deadline)
        
        if val < best_val:
            best_val = val
            best_result = x_opt.copy()
            
    # Try to polish the best result found with smaller steps
    if time.time() < deadline - 5.0:
         # We can try running more iterations on the best result
         # But _optimize_sequence already does this. 
         # Maybe try a different initial condition based on best_result?
         pass

    return best_result

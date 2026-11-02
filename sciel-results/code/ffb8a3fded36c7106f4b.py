import numpy as np
import time

N = 400

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
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

def _subgrad_max_abs_ac(x):
    """
    Compute the subgradient of max(|ac_k|) at x.
    Returns the gradient vector and the index of the max lag.
    """
    ac = _autoconv_fft(x)
    abs_ac = np.abs(ac)
    k_star = np.argmax(abs_ac)
    sign_ac = np.sign(ac[k_star])
    
    # Gradient of ac_k_star w.r.t x
    # ac_k = sum_j x_j x_{k-j}
    # d(ac_k)/dx_j = x_{k-j} + x_{j-k}
    
    grad = np.zeros(N)
    # Vectorized computation of gradient
    # For j in 0..N-1:
    # term1: x[k_star - j] if 0 <= k_star - j < N
    # term2: x[j - k_star] if 0 <= j - k_star < N
    
    # We can use np.roll or slicing, but simple vectorized logic is safer for bounds
    # Let's create indices
    idx = np.arange(N)
    
    # Term 1: x[k_star - idx]
    idx1 = k_star - idx
    mask1 = (idx1 >= 0) & (idx1 < N)
    grad1 = np.zeros(N)
    grad1[mask1] = x[idx1[mask1]]
    
    # Term 2: x[idx - k_star]
    idx2 = idx - k_star
    mask2 = (idx2 >= 0) & (idx2 < N)
    grad2 = np.zeros(N)
    grad2[mask2] = x[idx2[mask2]]
    
    grad = grad1 + grad2
    grad *= sign_ac
    
    return grad, k_star, ac[k_star]

def construct():
    START_TIME = time.time()
    TIME_LIMIT = 140  # safety margin before 150s kill

    # Normalize initial guess to sum=1
    best_x = np.ones(N)
    best_x /= np.sum(best_x)
    best_val = _objective(best_x)
    
    rng = np.random.default_rng(42)
    
    # Try a few diverse initializations normalized to sum=1
    initializations = [
        np.ones(N),
        np.linspace(1.0, 2.0, N),
        np.linspace(2.0, 1.0, N),
        0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, N))),
        0.5 * (1 + np.sin(2 * np.pi * np.linspace(0, 1, N))) + 0.5,
        np.exp(-np.linspace(0, 2, N)),
    ]
    
    for x0 in initializations:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        s = np.sum(x0)
        if abs(s) < 1e-6: continue
        x0_norm = x0 / s
        val = _objective(x0_norm)
        if val < best_val:
            best_val = val
            best_x = x0_norm.copy()

    # Nesterov's Accelerated Gradient for min max(|ac|) s.t. sum(x)=1
    # Objective: f(x) = max_k |ac_k(x)|
    # Constraint: sum(x) = 1
    
    x = best_x.copy()
    y = x.copy()
    t_k = 1.0
    momentum = 0.9
    
    # Step size decay
    # For convex functions, step size ~ 1/sqrt(t) is standard for subgrad,
    # but for Nesterov on smooth approximations it's 1/t.
    # Since this is non-smooth, we use a robust step size.
    # We'll use a diminishing step size with restarts if stuck.
    
    base_lr = 0.5
    for it in range(50000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Compute subgradient at y
        grad_y, k_star, val_y = _subgrad_max_abs_ac(y)
        
        # Update t_k
        t_k_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
        beta = (t_k - 1) / t_k_new
        
        # Step size
        # Heuristic: scale by 1/sqrt(it+1) or constant with restart
        lr = base_lr / np.sqrt(it + 1)
        
        # Gradient step
        x_new = y - lr * grad_y
        
        # Project to sum=1
        s_new = np.sum(x_new)
        if abs(s_new) < 1e-10:
             x_new = x_new + 1e-6 * rng.normal(size=N)
             s_new = np.sum(x_new)
        x_new /= s_new
        
        # Nesterov momentum update
        y_new = x_new + beta * (x_new - x)
        
        # Project y_new to sum=1 as well to stay on manifold
        s_y = np.sum(y_new)
        if abs(s_y) < 1e-10:
             y_new = y_new + 1e-6 * rng.normal(size=N)
             s_y = np.sum(y_new)
        y_new /= s_y
        
        x = x_new
        y = y_new
        t_k = t_k_new
        
        # Check improvement
        val_x = _objective(x)
        if val_x < best_val:
            best_val = val_x
            best_x = x.copy()
            
        # Periodic restart/randomization to escape local minima (though convex, numerical issues or non-unique max can cause issues)
        if it % 1000 == 0 and it > 0:
             # Small random perturbation
             x += 0.01 * rng.normal(size=N)
             s = np.sum(x)
             x /= s
             y = x.copy()
             t_k = 1.0

    return best_x

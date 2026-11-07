import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_value(h):
    """
    Computes the value as defined in the task.
    h: array of 50 non-negative floats.
    """
    # Ensure h is 1D array
    h = np.asarray(h, dtype=float)
    
    # Convolution
    g = np.convolve(h, h)
    m = len(g)
    
    # y is g padded with zeros
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Points
    # np.linspace(-0.5, 0.5, m + 2)
    # Spacing w
    w = 1.0 / (m + 1)
    
    # l2 calculation
    # sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # Note: indices 0 to m inclusive for the sum terms involving y[i] and y[i+1]
    # y has length m+2, so y[0]..y[m+1] are valid.
    # The sum is for i=0 to m.
    # Terms: y[i]^2 + y[i]y[i+1] + y[i+1]^2
    
    # Vectorized calculation
    yi = y[:-1]  # length m+1
    yip1 = y[1:] # length m+1
    terms = yi**2 + yi * yip1 + yip1**2
    l2 = (w / 3.0) * np.sum(terms)
    
    # Norms
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # We want to maximize value.
    # Value is scale invariant. We can fix sum(h) = 1.
    
    def objective(params):
        # params are log-heights or similar to ensure positivity
        # Let h = exp(params) / sum(exp(params)) to enforce sum(h)=1 and h>0
        # However, exp can overflow. Let's use softmax or just normalize exp.
        # If params are large, exp overflows.
        # Better: h_i = exp(params_i - max(params)) / sum(exp(params - max(params)))
        
        p = np.asarray(params)
        p_max = np.max(p)
        h_unnorm = np.exp(p - p_max)
        h = h_unnorm / np.sum(h_unnorm)
        
        val = compute_value(h)
        return -val  # Minimize negative value
    
    # Initial guesses
    # 1. Uniform
    h_uniform = np.ones(N)
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    h_gauss = np.exp(-x**2)
    # 3. Cosine
    h_cos = np.cos(np.pi * x) ** 2
    h_cos[h_cos < 0] = 0
    # 4. Triangular
    h_tri = 1 - np.abs(x)
    
    initial_guesses = [h_uniform, h_gauss, h_cos, h_tri]
    
    best_val = -np.inf
    best_h = None
    
    # Convert initial guesses to log-space for optimization
    # We need to invert the normalization to get initial params.
    # Since the objective normalizes internally, we can just pass log(h) as initial params.
    # But if h has zeros, log is -inf. Add small epsilon.
    
    for h_init in initial_guesses:
        if time.time() > deadline:
            break
            
        # Normalize h_init
        h_init = h_init / np.sum(h_init)
        h_init_safe = h_init + 1e-10
        params_init = np.log(h_init_safe)
        
        # Run optimization
        # Using Nelder-Mead for robustness with non-smooth boundaries (though we are in interior)
        # Or L-BFGS-B. L-BFGS-B is faster for smooth problems.
        # The objective is smooth with respect to params.
        
        try:
            res = minimize(
                objective, 
                params_init, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_val:
                best_val = res.fun
                # Reconstruct h from res.x
                p = res.x
                p_max = np.max(p)
                h_opt = np.exp(p - p_max)
                h_opt = h_opt / np.sum(h_opt)
                best_h = h_opt
        except Exception:
            pass
            
    # If optimization failed or didn't improve, use the best initial guess
    if best_h is None:
        best_h = h_uniform
    else:
        # Verify the value of the optimized h
        val_check = compute_value(best_h)
        if val_check > -best_val: # -best_val is the maximized value
            pass # It's consistent
        else:
            # Fallback to best initial
            for h_init in initial_guesses:
                h_init = h_init / np.sum(h_init)
                v = compute_value(h_init)
                if v > compute_value(best_h):
                    best_h = h_init

    # Final check: ensure non-negative and finite
    best_h = np.asarray(best_h, dtype=float)
    best_h[best_h < 0] = 0
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(f"Score: {compute_value(h)}")

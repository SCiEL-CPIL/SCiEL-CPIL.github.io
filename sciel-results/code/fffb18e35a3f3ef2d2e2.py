import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
X_MIN = -0.25
X_MAX = 0.25

# EVOLVE-BLOCK-START
def _compute_ratio_gaussian(params):
    """
    Compute max(autoconv) / l2_sq for a function defined by Gaussian superposition.
    params = [a1, x1, s1, a2, x2, s2, ...] where a>0, x in [-0.25, 0.25], s>0.
    """
    M = 5 # Number of Gaussians
    # Unpack parameters
    # To ensure positivity of amplitude and sigma, we can use raw params and transform,
    # or just bound them in the optimizer. Here we assume bounds are handled by optimizer,
    # but we need to be careful with sigma -> 0.
    
    a = params[0::3]
    x = params[1::3]
    s = params[2::3]
    
    # Ensure sigma is not too small to avoid numerical issues, though bounds should help
    s = np.clip(s, 1e-4, None)
    
    # Construct fine grid for evaluation
    n_grid = 2048
    x_fine = np.linspace(X_MIN, X_MAX, n_grid, endpoint=False)
    
    h_fine = np.zeros(n_grid, dtype=np.float64)
    for j in range(M):
        h_fine += a[j] * np.exp(-((x_fine - x[j])**2) / (2 * s[j]**2))
        
    # Enforce non-negativity (just in case negative amplitudes sneak in, though bounds prevent it)
    h_fine = np.clip(h_fine, 0.0, None)
    
    if np.sum(h_fine) < 1e-15:
        return 1e10
        
    # Compute autoconvolution
    n_fft = 4096
    F = np.fft.rfft(h_fine, n_fft)
    conv = np.fft.irfft(F * F, n_fft)[:2 * n_grid - 1]
    dx = 0.5 / n_grid
    conv *= dx
    
    max_conv = np.max(conv)
    l2_sq = np.sum(h_fine ** 2) * dx
    
    if l2_sq < 1e-15:
        return 1e10
        
    return max_conv / l2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    M = 5 # Number of Gaussians
    n_params = 3 * M
    
    # Bounds:
    # Amplitude a: [0, 5] (scaling doesn't matter for ratio, but helps numerical stability)
    # Center x: [-0.25, 0.25]
    # Sigma s: [0.001, 0.3]
    
    bounds = []
    for _ in range(M):
        bounds.append((0.0, 5.0))
        bounds.append((X_MIN, X_MAX))
        bounds.append((0.001, 0.3))
        
    best_ratio = 1.0 # Uniform baseline
    best_params = None
    
    # Initial guesses
    # 1. Uniform-like: Wide Gaussian
    # 2. Random starts
    
    starts = []
    
    # Start 1: Single wide Gaussian approximating uniform
    a0 = np.ones(M) * 1.0
    x0 = np.zeros(M)
    s0 = np.ones(M) * 0.2
    params0 = np.zeros(n_params)
    params0[0::3] = a0
    params0[1::3] = x0
    params0[2::3] = s0
    starts.append(params0)
    
    # Start 2: Symmetric pair
    if M >= 2:
        params1 = np.zeros(n_params)
        # Two Gaussians at +/- 0.1
        params1[0] = 1.0
        params1[1] = -0.1
        params1[2] = 0.1
        params1[3] = 1.0
        params1[4] = 0.1
        params1[5] = 0.1
        # Others zero or small
        starts.append(params1)

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(20):
        a_rand = rng.uniform(0.1, 2.0, M)
        x_rand = rng.uniform(X_MIN, X_MAX, M)
        s_rand = rng.uniform(0.01, 0.2, M)
        p_rand = np.zeros(n_params)
        p_rand[0::3] = a_rand
        p_rand[1::3] = x_rand
        p_rand[2::3] = s_rand
        starts.append(p_rand)
        
    # Optimize from each start
    for i, p_start in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(
                _compute_ratio_gaussian, 
                p_start, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            
            if res.fun < best_ratio:
                best_ratio = res.fun
                best_params = res.x
                
        except Exception as e:
            pass
            
    # If no improvement, use uniform
    if best_params is None:
        # Return uniform step function
        return [1.0] * N
        
    # Construct the final step function
    # Sample h(x) at the midpoints of the 600 intervals
    x_mids = -0.25 + (np.arange(N) + 0.5) * W
    
    a = best_params[0::3]
    x_c = best_params[1::3]
    s = best_params[2::3]
    s = np.clip(s, 1e-4, None)
    
    h_steps = np.zeros(N, dtype=np.float64)
    for j in range(M):
        h_steps += a[j] * np.exp(-((x_mids - x_c[j])**2) / (2 * s[j]**2))
        
    h_steps = np.clip(h_steps, 0.0, None)
    
    return h_steps.tolist()

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconvolution_peak_ratio(x):
    """
    Compute the maximum absolute value of the autoconvolution divided by the square of the integral.
    For a sequence x of length N, the autoconvolution is conv(x, x[::-1]) or similar.
    Here, we interpret 'autoconvolution' as the standard convolution of x with itself.
    The integral is sum(x).
    Score = max(|conv(x, x)|) / (sum(x)^2)
    """
    if len(x) == 0:
        return 0.0
    s = np.sum(x)
    if np.abs(s) < 1e-12:
        return float('inf')
    conv = np.convolve(x, x, mode='full')
    return np.max(np.abs(conv)) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Initialize with a random signed sequence
    rng = np.random.default_rng(42)
    x = rng.choice([-1.0, 1.0], size=N)
    
    current_score = _autoconvolution_peak_ratio(x)
    best_x = x.copy()
    best_score = current_score
    
    # Gradient-free local search using Nelder-Mead on a subset of coordinates?
    # Or use a simple coordinate descent / flip search.
    # Since the landscape is discrete, a stochastic local search with simulated annealing might work.
    # However, N=400 is small enough for efficient numpy operations.
    
    # Let's try a few random restarts and local flips.
    num_restarts = 50
    flips_per_iter = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        if r == 0:
            x = best_x.copy()
        else:
            x = rng.choice([-1.0, 1.0], size=N)
        
        current_score = _autoconvolution_peak_ratio(x)
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
        
        # Local search: flip random elements and accept if it improves
        for i in range(flips_per_iter):
            if time.time() > deadline:
                break
            
            # Try flipping one random element
            idx = rng.integers(0, N)
            x_old = x[idx]
            x[idx] = -x_old
            
            new_score = _autoconvolution_peak_ratio(x)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                x[idx] = x_old
    
    # Final polish: try flipping every single element once more to ensure local optimum
    if time.time() < deadline - 5.0:
        for idx in range(N):
            if time.time() > deadline:
                break
            x_old = best_x[idx]
            best_x[idx] = -x_old
            new_score = _autoconvolution_peak_ratio(best_x)
            if new_score < best_score:
                best_score = new_score
            else:
                best_x[idx] = x_old
    
    return best_x
# EVOLVE-BLOCK-END

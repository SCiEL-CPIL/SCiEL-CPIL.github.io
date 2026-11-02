import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autocorr_max_abs(x):
    """Compute max absolute value of circular autocorrelation using FFT."""
    X = np.fft.rfft(x)
    power = X * np.conj(X)
    acf = np.fft.irfft(power, N)
    return np.max(np.abs(acf))

def _compute_score(x):
    """
    Compute the score: max|autoconvolution| / (integral)^2.
    """
    acf_max = _autocorr_max_abs(x)
    total_sum = np.sum(x)
    norm_sq = total_sum * total_sum
    if norm_sq < 1e-12:
        return np.inf
    return acf_max / norm_sq

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_x = np.ones(N)
    best_score = _compute_score(best_x)
    
    # Objective function for continuous relaxation
    # We minimize f(x) = max|autoconv| / (sum x)^2
    # Using a soft-max approximation for differentiability
    def objective(x_cont):
        if time.time() > deadline:
            return np.inf
        x_float = np.tanh(x_cont) # Map R^N to (-1, 1)^N
        # Compute score
        acf_max = _autocorr_max_abs(x_float)
        total_sum = np.sum(x_float)
        norm_sq = total_sum * total_sum
        if norm_sq < 1e-12:
            return 1e6
        return acf_max / norm_sq

    # Gradient is expensive to compute analytically for max|acf|, 
    # so we rely on numerical differentiation or quasi-Newton methods 
    # with finite differences if analytic grad is too complex. 
    # However, L-BFGS-B supports finite differences.
    
    # Multi-start strategy
    rng = np.random.RandomState(42)
    
    # Start 1: All ones (mapped to infinity in tanh pre-image, but we start near 0 for tanh(0)=0? 
    # Wait, tanh(0)=0. We want to start near x=1. 
    # If x_cont is large positive, tanh(x_cont) ~ 1.
    # Let's initialize x_cont such that tanh(x_cont) ~ 1.
    # Or simpler: just optimize x_float directly with bounds? 
    # No, tanh ensures we stay in (-1,1). 
    # To start at x=1, x_cont should be large. Let's use x_cont = 5.0.
    
    # Actually, a better approach: Optimize the discrete bits via a continuous surrogate 
    # that allows partial flips, or just use the continuous relaxation to find a good 
    # "soft" assignment, then round.
    
    # Let's try optimizing x_cont where x_float = tanh(x_cont).
    # Initial x_cont = 5.0 (tanh(5) ~ 0.9999)
    x0 = np.full(N, 5.0)
    
    # We will run a few random restarts with different initializations
    # to escape local minima.
    
    for restart in range(3):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x0 = np.full(N, 5.0)
        else:
            # Random initialization
            x0 = rng.uniform(-5, 5, N)
            
        # Run L-BFGS-B
        # We don't provide analytic gradient, so it will use numerical differentiation.
        # This might be slow. Let's see if we can afford it.
        # N=400. Finite difference grad requires N+1 evaluations.
        # Each evaluation is an FFT (O(N log N)).
        # So one grad eval is ~401 * O(N log N).
        # L-BFGS-B might take 50-100 iterations.
        # Total evals ~ 100 * 400 = 40,000 FFTs.
        # FFT of size 400 is fast. 40,000 * 10us = 0.4s. Should be fine.
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'disp': False})
        except Exception:
            continue
            
        if time.time() > deadline:
            break
            
        x_opt_cont = res.x
        x_float = np.tanh(x_opt_cont)
        
        # Round to nearest integer {-1, 1}
        x_rounded = np.sign(x_float)
        # Handle zeros (unlikely with tanh)
        x_rounded[x_rounded == 0] = 1
        
        # Local search refinement on the rounded discrete vector
        # Use a few steps of hill climbing to fix any rounding errors
        x_cur = x_rounded.copy()
        cur_score = _compute_score(x_cur)
        
        # Simple hill climb with limited steps
        steps = 500
        for i in range(steps):
            if time.time() > deadline:
                break
            idx = rng.randint(N)
            old_val = x_cur[idx]
            x_cur[idx] = -old_val
            new_score = _compute_score(x_cur)
            if new_score < cur_score:
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x_cur.copy()
            else:
                x_cur[idx] = old_val
                
    return best_x

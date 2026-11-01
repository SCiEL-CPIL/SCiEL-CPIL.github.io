"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s kill

    def max_autocorr(h):
        """
        Compute the maximum absolute autocorrelation at non-zero lags.
        """
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft).real[:N]
        return np.max(np.abs(acf[1:]))

    def continuous_objective(x):
        """
        Objective for continuous optimization: minimize max autocorrelation.
        x is a real vector of length N. We project to signed values via tanh.
        """
        h = np.tanh(x)
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft).real[:N]
        # Use a smooth approximation of max to help gradient methods
        # Actually, let's just use the raw max for the objective, but the gradient
        # will be handled by scipy's numerical differentiation or we can provide it.
        # For L-BFGS-B, numerical gradients are fine for N=400.
        return np.max(np.abs(acf[1:]))

    def discrete_objective(h):
        """
        Objective for discrete local search: h is a ±1 vector.
        """
        return max_autocorr(h)

    def local_search_discrete(h, max_iters=5000):
        """
        Greedy local search: flip bits that reduce the objective.
        """
        current_h = h.copy()
        current_val = discrete_objective(current_h)
        
        # Precompute contributions? For N=400, O(N^2) per eval is 160k ops.
        # FFT is ~N log N. Let's use FFT for speed.
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                break
            
            best_gain = 0
            best_idx = -1
            
            # Randomly shuffle indices to break symmetry
            indices = np.random.permutation(N)
            
            for i in indices:
                # Try flipping i
                old_val = current_h[i]
                current_h[i] = -old_val
                new_val = discrete_objective(current_h)
                gain = current_val - new_val
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_new_val = new_val
                else:
                    current_h[i] = old_val # revert
            
            if best_idx >= 0:
                current_h[best_idx] = -current_h[best_idx]
                current_val = best_new_val
                improved = True
        
        return current_h, current_val

    # Strategy:
    # 1. Run several continuous optimizations from different random starts.
    # 2. Project results to ±1.
    # 3. Refine with discrete local search.
    
    rng = np.random.RandomState(123)
    
    best_h = None
    best_val = np.inf
    
    num_starts = 5
    
    for s in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Random start in continuous space
        x0 = rng.randn(N) * 0.5
        
        # Optimize
        res = minimize(
            continuous_objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'maxfun': 50000, 'maxls': 50}
        )
        
        # Project to discrete
        h_proj = np.sign(res.x)
        # Handle zeros (unlikely with tanh, but sign(0)=0)
        h_proj[h_proj == 0] = 1.0
        
        # Discrete refinement
        h_refined, val_refined = local_search_discrete(h_proj, max_iters=2000)
        
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined
        
        # Also try the unrefined projection just in case local search got stuck early
        val_proj = discrete_objective(h_proj)
        if val_proj < best_val:
            best_val = val_proj
            best_h = h_proj.copy()
            
        # Check time
        if time.time() > deadline - 10:
            break

    # If we have time, do one more round of discrete local search on the best
    if best_h is not None and time.time() < deadline - 5:
        h_final, val_final = local_search_discrete(best_h, max_iters=5000)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final

    # Fallback: if something went wrong or no improvement, return a random sequence
    if best_h is None:
        best_h = rng.choice([1, -1], N).astype(np.float64)

    return best_h
# EVOLVE-BLOCK-END

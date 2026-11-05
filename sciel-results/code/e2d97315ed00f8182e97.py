"""Program using Simulated Annealing with normalization to reduce the third autocorrelation constant."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _objective(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e12
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s * s))

def construct():
    # Initialize with random positive values to avoid zero sum initially
    # Sum=1 normalization is a good idea to keep the scale fixed.
    # Let's work in the space of vectors that sum to 1.
    # If we perturb h, we re-normalize.
    
    # Start with a "white noise" like positive vector
    rng = np.random.default_rng(123)
    h = rng.uniform(0.5, 1.5, N)
    h = h / np.sum(h) # Normalize to sum 1
    
    current_val = _objective(h)
    best_h = h.copy()
    best_val = current_val
    
    # Simulated Annealing parameters
    T_start = 0.1
    T_end = 1e-6
    cooling_rate = 0.999
    steps_per_temp = 10
    max_temps = 5000 # 5000 * 10 = 50000 steps
    
    T = T_start
    for t in range(max_temps):
        # Check time or just run fixed iterations
        for _ in range(steps_per_temp):
            # Generate a perturbation
            # Randomly pick a few indices to perturb (block update)
            n_perturb = rng.integers(1, 5)
            indices = rng.choice(N, size=n_perturb, replace=False)
            
            h_new = h.copy()
            # Add Gaussian noise
            noise = rng.normal(0, 0.1, n_perturb)
            h_new[indices] += noise
            
            # Ensure finite
            if not np.all(np.isfinite(h_new)):
                continue
                
            # Normalize to sum 1
            s_new = np.sum(h_new)
            if abs(s_new) < 1e-10:
                # Reject if sum is zero
                continue
            h_new = h_new / s_new
            
            val_new = _objective(h_new)
            
            # Acceptance criterion
            delta = val_new - current_val
            if delta < 0 or (T > 0 and np.random.rand() < np.exp(-delta / T)):
                h = h_new
                current_val = val_new
                
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
        
        T *= cooling_rate
        if T < T_end:
            break
            
    # Final local polish with simple hill climbing on the best found
    # to squeeze out the last bit of improvement
    h = best_h.copy()
    current_val = best_val
    step_size = 0.05
    for _ in range(500):
        improved = False
        idx = rng.integers(N)
        # Try small perturbations
        for _ in range(5):
            delta_h = rng.uniform(-step_size, step_size)
            h_try = h.copy()
            h_try[idx] += delta_h
            s_try = np.sum(h_try)
            if abs(s_try) < 1e-10:
                continue
            h_try = h_try / s_try
            val_try = _objective(h_try)
            if val_try < current_val:
                h = h_try
                current_val = val_try
                best_val = val_try
                best_h = h.copy()
                improved = True
                break
        if not improved:
            step_size *= 0.95
            if step_size < 1e-6:
                break

    return best_h
# EVOLVE-BLOCK-END

import numpy as np
import random
import time

N = 400

def construct():
    def get_val(h):
        s = np.sum(h)
        if s == 0:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        return abs(2 * N * m / (s * s))
    
    # Simulated Annealing Parameters
    T_start = 100.0
    T_end = 1e-4
    cooling_rate = 0.995
    max_iterations = 5000
    
    # We will try a few different starting points
    start_points = [
        np.ones(N),
        np.random.default_rng(1).normal(size=N),
        np.random.default_rng(2).normal(size=N) * 0.1 + 1.0
    ]
    
    best_h = np.ones(N)
    best_val = get_val(best_h)
    
    for h0 in start_points:
        h = h0.copy()
        current_val = get_val(h)
        T = T_start
        
        # Adaptive step size
        step_size = 1.0
        
        for i in range(max_iterations):
            # Generate a new candidate
            # Perturb a random subset of indices to allow local fine-tuning
            # or perturb all with small noise. Let's perturb all with Gaussian noise.
            # The magnitude of noise is scaled by step_size and T.
            noise = np.random.normal(0, step_size * (T / T_start + 0.1), size=N)
            h_new = h + noise
            
            # Ensure finite
            if not np.all(np.isfinite(h_new)):
                h_new = h
                new_val = current_val
            else:
                new_val = get_val(h_new)
            
            # Acceptance criterion
            if new_val < current_val:
                h = h_new
                current_val = new_val
                # Reduce step size slightly if we improved
                step_size *= 0.99
            else:
                # Accept with probability exp(-delta/T)
                delta = new_val - current_val
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        h = h_new
                        current_val = new_val
                        # Increase step size if we accepted a worse solution (exploration)
                        step_size *= 1.01
                else:
                    # This shouldn't happen if new_val >= current_val, but just in case
                    h = h_new
                    current_val = new_val
            
            # Update global best
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
            
            # Cool down
            T *= cooling_rate
            
            # Cap step size to prevent divergence
            if step_size > 10.0:
                step_size = 10.0
            if step_size < 1e-6:
                step_size = 1e-6
                
    return best_h

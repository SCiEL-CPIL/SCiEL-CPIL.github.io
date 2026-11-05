import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 140s limit
    
    # Initialize with random walk or Gaussian
    # The hint suggests random walk with negative drift
    # Let's try a few initializations
    
    best_h = None
    best_val = float('inf')
    
    # Helper to compute value
    def compute_val(h):
        s = np.sum(h)
        if np.isclose(s, 0):
            return float('inf')
        # Use FFT for speed
        # Zero pad to 2N-1
        n_fft = 2 * N
        H = np.fft.fft(h, n=n_fft)
        C = np.fft.ifft(H * H).real
        # The convolution result is in C
        # Max of signed entries
        max_c = np.max(C)
        val = abs(2 * N * max_c / (s ** 2))
        return val

    # Try a few random starts
    rng = np.random.default_rng(42)
    
    for _ in range(10):
        # Random Walk with drift
        steps = rng.normal(0, 1, N)
        h = np.cumsum(steps)
        # Add negative drift to make sum close to 0? 
        # Actually we normalize, so drift doesn't matter for the value, 
        # but it might affect the landscape.
        # Let's just use Gaussian noise for simplicity first, 
        # then refine.
        h = rng.normal(0, 1, N)
        
        # Normalize
        s = np.sum(h)
        if s != 0:
            h = h / s
            
        val = compute_val(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    # Simulated Annealing
    if best_h is not None:
        h = best_h.copy()
        current_val = best_val
        T = 1.0
        T_min = 1e-6
        cooling = 0.999
        step_size = 0.1
        
        while time.time() < deadline and T > T_min:
            # Pick a random index
            i = rng.integers(0, N)
            
            # Perturb
            h_new = h.copy()
            h_new[i] += rng.normal(0, step_size)
            
            # Normalize
            s_new = np.sum(h_new)
            if np.isclose(s_new, 0):
                T *= cooling
                continue
            h_new = h_new / s_new
            
            new_val = compute_val(h_new)
            
            # Accept if better or with prob exp(-(new-old)/T)
            if new_val < current_val:
                h = h_new
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                delta = new_val - current_val
                if delta < 0:
                    delta = 0 # avoid log(0) if somehow negative due to float err? No, new>old
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    h = h_new
                    current_val = new_val
            
            T *= cooling
            # Adapt step size if stuck?
            # Simple version: keep constant or decay
    
    return best_h

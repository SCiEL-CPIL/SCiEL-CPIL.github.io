import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Initial Guess: Rudin-Shapiro
    # RS sequence generation
    p = [1]
    q = [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    coeffs = np.array(p[:24], dtype=np.float64)
    
    # Evaluation function
    def get_peak(c):
        # Use FFT for fast evaluation on a grid
        M = 8192
        # Zero pad to M
        c_padded = np.zeros(M)
        c_padded[:len(c)] = c
        # FFT
        Y = np.fft.fft(c_padded)
        # Magnitude
        mag = np.abs(Y)
        # Find max
        max_idx = np.argmax(mag)
        max_val_grid = mag[max_idx]
        
        # Refine the peak
        # The peak is at theta = 2*pi*max_idx/M
        # We can do a local search on theta
        # f(theta) = |P(e^{i theta})|
        # P(z) = sum c_k z^k
        # We can evaluate P exactly for a specific theta
        
        def peak_func(theta):
            z = np.exp(1j * theta)
            # Evaluate polynomial
            # Use Horner's rule or direct sum
            # Direct sum with vectorization
            vals = np.sum(c * np.exp(1j * np.arange(len(c)) * theta))
            return np.abs(vals)
            
        # Golden section search around the grid peak
        theta0 = 2 * np.pi * max_idx / M
        # Search in a small window, e.g., +/- 2 grid steps
        delta = 2 * np.pi / M
        # Minimize -peak_func to maximize peak_func
        # Or just use minimize_scalar on -peak_func
        
        # To speed up, we can just check a few points around the grid peak
        # Or use a simple hill climbing on theta
        
        # Let's use a small grid refinement
        # Points around theta0
        thetas = theta0 + np.linspace(-2*delta, 2*delta, 100)
        peaks = np.array([peak_func(t) for t in thetas])
        local_max_idx = np.argmax(peaks)
        local_max_val = peaks[local_max_idx]
        
        # Further refinement with golden section if needed?
        # For N=24, the 100-point grid around the peak is likely sufficient for comparison
        # unless the peak is very sharp.
        
        return local_max_val

    def flip(c, idx):
        c_new = c.copy()
        c_new[idx] *= -1
        return c_new

    best_coeffs = coeffs.copy()
    best_score = get_peak(best_coeffs)
    
    # Local Search
    def local_search(c):
        improved = True
        while improved:
            improved = False
            for i in range(len(c)):
                c_new = flip(c, i)
                score = get_peak(c_new)
                if score < best_score: # Wait, best_score should be the current c's score?
                    # No, local search should minimize the peak of the current candidate
                    pass
        
        # Let's rewrite local search properly
        current_score = get_peak(c)
        improved = True
        while improved:
            improved = False
            for i in range(len(c)):
                c_new = flip(c, i)
                new_score = get_peak(c_new)
                if new_score < current_score:
                    c = c_new
                    current_score = new_score
                    improved = True
        return c, current_score

    best_coeffs, best_score = local_search(best_coeffs)
    
    # Simulated Annealing
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random index to flip
        i = np.random.randint(len(current_coeffs))
        new_coeffs = flip(current_coeffs, i)
        new_score = get_peak(new_coeffs)
        
        delta = new_score - current_score
        
        if delta < 0:
            current_coeffs = new_coeffs
            current_score = new_score
            if current_score < best_score:
                best_coeffs = current_coeffs.copy()
                best_score = current_score
        else:
            # Accept with probability exp(-delta / T)
            if np.random.rand() < np.exp(-delta / T):
                current_coeffs = new_coeffs
                current_score = new_score
                
        T *= cooling
        
    # Final Local Search on best found
    best_coeffs, best_score = local_search(best_coeffs)
    
    # Convert back to list of ints
    return [1 if c > 0 else -1 for c in best_coeffs]

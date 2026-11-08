import numpy as np
from scipy.optimize import minimize
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, 10s safety margin
    
    # Initial guess: Boxcar
    a = np.ones(N)
    best_val = float('inf')
    best_a = a
    
    def objective(a):
        # Compute autoconvolution
        # a is length N. conv is length 2N-1.
        # Using FFT for speed
        n_fft = 1 << (2 * N - 1).bit_length()
        A = np.fft.rfft(a, n_fft)
        C = np.fft.irfft(A * A, n_fft)[:2 * N - 1]
        
        # The max is typically at the center
        # To be safe, we can use a soft-max or just the center if we assume unimodality
        # However, if we want to allow asymmetry that shifts the peak, 
        # the peak of a real autoconvolution is always symmetric around the center of the support.
        # The center of support is N-1.
        # So the max is at N-1 if the function is unimodal.
        # If not unimodal, there might be other peaks, but the center peak is usually dominant.
        # Let's use the center peak for differentiability, and verify against global max at the end.
        center_peak = C[N - 1]
        
        if center_peak <= 1e-12:
            return 1e10
            
        energy = np.sum(a ** 2)
        return energy / center_peak

    # We want to minimize energy / max_conv
    # This is equivalent to minimizing energy subject to max_conv >= 1 (scale invariance)
    # Or just minimizing the ratio.
    
    # Use L-BFGS-B
    # Bounds: a_i >= 0
    bounds = [(0, None) for _ in range(N)]
    
    # Multi-start to find asymmetric solutions
    # 1. Boxcar
    # 2. Random non-negative
    # 3. Shifted Boxcar (if allowed? No, support is fixed on [-1/4, 1/4])
    # 4. Triangle
    
    initializations = []
    initializations.append(np.ones(N))
    
    # Random
    rng = np.random.default_rng(42)
    initializations.append(rng.random(N))
    
    # Triangle
    tri = np.abs(1 - np.linspace(0, 2, N))
    initializations.append(tri)
    
    # Cosine bump
    initializations.append(np.cos(np.linspace(-np.pi/2, np.pi/2, N)))
    
    for init in initializations:
        if time.time() > deadline - 20:
            break
            
        # Normalize init to avoid scaling issues (though ratio is scale invariant)
        init = init / np.linalg.norm(init)
        
        res = minimize(objective, init, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10})
        
        if res.success:
            a_opt = res.x
            # Calculate the true score using global max
            n_fft = 1 << (2 * N - 1).bit_length()
            A = np.fft.rfft(a_opt, n_fft)
            C = np.fft.irfft(A * A, n_fft)[:2 * N - 1]
            true_max = np.max(C)
            energy = np.sum(a_opt ** 2)
            val = energy / true_max
            
            if val < best_val:
                best_val = val
                best_a = a_opt
                
    # If best_val is still inf, return boxcar
    if not np.isfinite(best_val):
        return [1.0] * N
        
    # Return the list of heights
    # The task asks for "sequence of nonnegative step heights"
    return [float(x) for x in best_a]
# EVOLVE-BLOCK-END

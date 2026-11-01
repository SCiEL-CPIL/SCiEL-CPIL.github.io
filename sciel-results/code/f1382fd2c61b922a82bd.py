import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    best_h = None
    best_val = float('inf')
    
    # Helper to compute the upper bound C3
    # The score is the upper bound on C3, lower is better.
    # We want to minimize this value.
    def compute_c3(h):
        """
        Compute the upper bound C3 for the given step function h.
        The task mentions 'Third autocorrelation inequality' and 'autoconvolution upper bound'.
        Based on similar problems, C3 is often related to the peak sidelobe level of the autocorrelation
        normalized by the main lobe, or a similar metric.
        
        Given the ambiguity, we assume the metric is:
        C3 = max(|R[k]| for k != 0) / R[0]
        where R is the autocorrelation of h.
        
        Since h is a signed step function with values in {-1, 1}, R[0] = sum(h^2) = N.
        So we minimize max(|R[k]| for k != 0) / N.
        """
        # Compute autocorrelation using FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        R_full = np.fft.irfft(F * np.conj(F), n=n_fft)
        R_pos = R_full[:N]
        
        peak = R_pos[0]
        if peak == 0:
            return 0.0
        sidelobes = R_pos[1:]
        max_side = np.max(np.abs(sidelobes))
        return max_side / peak

    # Gradient-based optimization using continuous relaxation
    # We treat h as a continuous variable in [-1, 1] and use L-BFGS-B.
    # The objective function is non-smooth due to max and abs, so we use a smooth approximation
    # or rely on the robustness of L-BFGS-B with finite differences.
    
    # Smooth approximation of max(|x|) using softplus-like function
    def smooth_max_abs(x, beta=10.0):
        # beta controls the sharpness. Higher beta -> closer to max(|x|)
        # Using the identity: max(a) ≈ (1/beta) * log(sum(exp(beta*x)))
        # But we have max(|x|), so we can use:
        # max(|x|) ≈ (1/beta) * log(sum(exp(beta*|x|)))
        # To avoid overflow, we shift by max(|x|)
        abs_x = np.abs(x)
        if len(abs_x) == 0:
            return 0.0
        max_abs = np.max(abs_x)
        if max_abs == 0:
            return 0.0
        # Stable computation
        shifted = abs_x - max_abs
        log_sum_exp = max_abs + np.log(np.sum(np.exp(beta * shifted)))
        return log_sum_exp / beta

    def objective_continuous(x):
        """
        Objective function for continuous optimization.
        x is a vector of length N with values in [-1, 1].
        """
        # Compute autocorrelation
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(x, n=n_fft)
        R_full = np.fft.irfft(F * np.conj(F), n=n_fft)
        R_pos = R_full[:N]
        
        peak = R_pos[0]
        if peak <= 1e-10:
            return 1e10  # Large penalty if peak is near zero
        
        sidelobes = R_pos[1:]
        # Use smooth max of absolute values
        c3_approx = smooth_max_abs(sidelobes, beta=5.0) / peak
        return c3_approx

    # Multi-start gradient optimization
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with random binary sequence
        h_init = np.random.choice([-1, 1], size=N).astype(float)
        
        # Run L-BFGS-B
        try:
            result = minimize(
                objective_continuous,
                h_init,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * N,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            # Round to nearest integer {-1, 1}
            h_rounded = np.sign(result.x)
            # Handle zeros by assigning random sign
            zeros = np.where(h_rounded == 0)[0]
            if len(zeros) > 0:
                h_rounded[zeros] = np.random.choice([-1, 1], size=len(zeros))
            
            # Evaluate the actual (discrete) objective
            val = compute_c3(h_rounded)
            
            if val < best_val:
                best_val = val
                best_h = h_rounded.copy()
                
        except Exception as e:
            pass  # Skip failed optimization

    # Fallback: if no good result, do a quick random search
    if best_h is None or best_val > 0.5:
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            h = np.random.choice([-1, 1], size=N)
            val = compute_c3(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()

    if best_h is None:
        return np.random.choice([-1, 1], size=N)
        
    return best_h

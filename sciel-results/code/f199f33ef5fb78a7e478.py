"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3_bound(heights):
    """Compute the upper bound on C3 from the autoconvolution of the step function."""
    # Step size
    dx = 1.0 / (2.0 * N)  # interval length is 1/2, N intervals
    
    # Autoconvolution: (f * f)(t) = sum_i f_i * f_j * dx where |i-j|*dx = t
    # For a step function, the autoconvolution at lag k (in steps) is:
    # sum_{i=0}^{N-1-k} f[i] * f[i+k] * dx for k >= 0
    # And it's symmetric: autoconv(-k) = autoconv(k)
    
    # Compute autoconvolution at all lags
    # autoconv[k] = sum_{i=0}^{N-1-|k|} f[i] * f[i+|k|] * dx
    # The maximum absolute value of this is our bound
    
    # Use FFT for efficiency
    # Zero-pad to avoid circular convolution
    n_fft = 2 * N
    f_padded = np.zeros(n_fft)
    f_padded[:N] = heights
    f_fft = np.fft.fft(f_padded)
    autoconv = np.fft.ifft(f_fft * np.conj(f_fft)).real
    autoconv *= dx  # scale by dx
    
    # The autoconvolution values at lags 0, 1, ..., N-1 are autoconv[0:N]
    # (and negative lags are the same by symmetry for real signals)
    # The maximum absolute value over all lags
    max_val = np.max(np.abs(autoconv[:N]))
    
    return max_val


def _optimize_heights(deadline):
    """Optimize the step heights to minimize the C3 bound."""
    start_time = time.time()
    
    # Start with alternating signs
    heights = np.ones(N)
    heights[1::2] = -1.0
    
    best_heights = heights.copy()
    best_bound = _compute_c3_bound(best_heights)
    
    # Try simple gradient-free optimization: coordinate descent with sign flips
    # For each position, try flipping the sign and see if it improves
    
    improved = True
    while improved and (time.time() - start_time) < deadline - 5.0:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > deadline - 5.0:
                break
            # Try flipping the sign at position i
            old_val = heights[i]
            heights[i] = -old_val
            new_bound = _compute_c3_bound(heights)
            if new_bound < best_bound:
                best_bound = new_bound
                best_heights = heights.copy()
                improved = True
            else:
                heights[i] = old_val
    
    # Try small perturbations around the best solution
    if (time.time() - start_time) < deadline - 5.0:
        # Local search: try adjusting individual heights by small amounts
        step_sizes = [0.1, 0.05, 0.01]
        for step in step_sizes:
            if (time.time() - start_time) > deadline - 5.0:
                break
            for i in range(N):
                if (time.time() - start_time) > deadline - 5.0:
                    break
                for delta in [-step, step]:
                    old_val = best_heights[i]
                    test_heights = best_heights.copy()
                    test_heights[i] = old_val + delta
                    test_bound = _compute_c3_bound(test_heights)
                    if test_bound < best_bound:
                        best_bound = test_bound
                        best_heights[i] = old_val + delta
    
    return best_heights, best_bound


def construct():
    """Construct the signed step heights that minimize the C3 autoconvolution upper bound."""
    deadline = 120.0  # seconds
    start_time = time.time()
    
    # Get optimized heights
    heights, bound = _optimize_heights(deadline - start_time)
    
    return heights
# EVOLVE-BLOCK-END


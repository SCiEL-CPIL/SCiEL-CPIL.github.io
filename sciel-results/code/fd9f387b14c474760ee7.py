"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _generate_m_sequence(n, poly_bits=8):
    """Generate n bits from an LFSR with primitive polynomial of degree poly_bits.
    Returns array of +1/-1 values."""
    # Use a known primitive polynomial for degree 8: x^8 + x^4 + x^3 + x^2 + 1
    # Taps: 8, 4, 3, 2, 1 (1-indexed, with x^8 as feedback)
    # In 0-indexed: taps at positions 7, 3, 2, 1, 0
    taps = [7, 3, 2, 1, 0]
    
    # Initialize LFSR state (avoid all zeros)
    state = np.ones(poly_bits, dtype=np.int8)
    
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        bits[i] = state[0]
        # Compute feedback
        feedback = 0
        for t in taps:
            feedback ^= state[t]
        # Shift right and insert feedback at position 0
        state = np.roll(state, -1)
        state[0] = feedback
    
    # Convert to +1/-1
    return 2 * bits - 1


def _compute_autocorr_bound(x):
    """Compute the upper bound on C3 from the autocorrelation of x.
    The bound is max over lags of |autocorr(lag)| for non-zero lags."""
    n = len(x)
    # Compute autocorrelation using FFT
    fft_x = np.fft.fft(x, n=2*n)
    autocorr = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
    # Non-zero lags: 1 to n-1
    non_zero_lags = autocorr[1:]
    return np.max(np.abs(non_zero_lags))


def _fast_flip_update(x, idx, current_bound):
    """Quickly estimate the new bound after flipping x[idx].
    Returns approximate new bound."""
    old_val = x[idx]
    x[idx] = -old_val
    
    # Recompute autocorrelation (full recompute for accuracy)
    n = len(x)
    fft_x = np.fft.fft(x, n=2*n)
    autocorr = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
    non_zero_lags = autocorr[1:]
    new_bound = np.max(np.abs(non_zero_lags))
    
    return new_bound


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # Initialize with m-sequence (LFSR)
    x = _generate_m_sequence(N, poly_bits=8)
    best_x = x.copy()
    best_bound = _compute_autocorr_bound(best_x)
    
    # Local search: try flipping individual bits
    np.random.seed(42)
    flip_order = np.random.permutation(N)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in flip_order:
            if time.time() > deadline:
                break
            
            old_val = best_x[idx]
            best_x[idx] = -old_val
            new_bound = _compute_autocorr_bound(best_x)
            
            if new_bound < best_bound:
                best_bound = new_bound
                improved = True
            else:
                best_x[idx] = old_val
        
        # If no improvement in this pass, try random restarts with small perturbations
        if not improved:
            # Try a few random multi-flip moves
            for _ in range(10):
                if time.time() > deadline:
                    break
                # Flip 2-3 random bits
                n_flips = np.random.randint(2, 4)
                flip_indices = np.random.choice(N, size=n_flips, replace=False)
                
                trial_x = best_x.copy()
                trial_x[flip_indices] *= -1
                trial_bound = _compute_autocorr_bound(trial_x)
                
                if trial_bound < best_bound:
                    best_x = trial_x
                    best_bound = trial_bound
                    improved = True
    
    return best_x
# EVOLVE-BLOCK-END

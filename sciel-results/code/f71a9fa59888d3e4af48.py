"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(h):
    """
    Compute the upper bound on C3, which is the sum of squares of the
    aperiodic autocorrelation coefficients (excluding lag 0).
    """
    # h is length N, values +/- 1
    # c[k] = sum_{j=0}^{N-1-k} h[j] h[j+k]
    # We want sum_{k=1}^{N-1} c[k]^2
    # Using FFT for efficiency:
    # Let H = FFT(h, 2N). The circular autocorrelation is IFFT(|H|^2).
    # The aperiodic autocorrelation c[k] for k>=0 is the first N elements of the
    # inverse FFT of |H|^2 (appropriately scaled? No, standard FFT autocorrelation
    # gives circular. For aperiodic, we zero-pad to 2N).
    
    # Actually, np.correlate is O(N^2) but N=400 is small.
    # However, FFT is O(N log N).
    # Let's use FFT.
    # Zero-pad h to length 2N to get linear convolution/autocorrelation.
    size = 2 * N
    H = np.fft.fft(h, n=size)
    # Power spectrum
    P = H * np.conj(H)
    # Inverse FFT gives circular autocorrelation of the zero-padded sequence.
    # Since the sequence is zero-padded to 2N, the first N lags of the circular
    # autocorrelation correspond to the aperiodic autocorrelation lags 0 to N-1.
    # Note: np.fft.ifft returns complex, but for real input it should be real.
    c = np.fft.ifft(P).real[:N]
    
    # c[0] is the sum of squares, which is N.
    # We want sum of c[k]^2 for k=1 to N-1.
    return np.sum(c[1:] ** 2)


def _compute_score_and_c(h):
    """
    Compute score and the autocorrelation vector c[1:N] for gradient calculation.
    """
    size = 2 * N
    H = np.fft.fft(h, n=size)
    P = H * np.conj(H)
    c_full = np.fft.ifft(P).real[:N]
    score = np.sum(c_full[1:] ** 2)
    return score, c_full


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s limit

    rng = np.random.RandomState(42)

    # Initialize with random signs
    best_h = rng.choice([-1.0, 1.0], size=N)
    best_score, _ = _compute_score_and_c(best_h)
    
    current_h = best_h.copy()
    current_score, current_c = best_score, _

    T = 1.0  # Initial temperature
    cooling = 0.9995

    # Precompute phases for potential FFT update? 
    # For N=400, full FFT is fast enough.
    # Let's just recompute score via FFT on each flip.
    # Or use the gradient method from inspiration 3?
    # Gradient method requires O(N^2) to compute G, which is 160,000 ops.
    # FFT is O(N log N) ~ 400 * 9 = 3600 ops.
    # So FFT recompute is likely faster than gradient calculation for N=400.
    # Let's stick to simple flip-and-evaluate with FFT.

    while time.time() < deadline:
        T *= cooling
        if T < 1e-8:
            T = 1e-8

        # Try a random flip
        i = rng.randint(N)
        old_val = current_h[i]
        
        # Flip
        current_h[i] = -old_val
        
        # Evaluate
        new_score, new_c = _compute_score_and_c(current_h)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or rng.rand() < np.exp(-delta_score / T):
            # Accept
            current_score = new_score
            current_c = new_c
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        else:
            # Reject
            current_h[i] = old_val

    return best_h
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _autocorr_peak(h):
    """Compute max absolute autocorrelation at non-zero lags."""
    n = len(h)
    if n <= 1:
        return 0.0
    # Use FFT for efficient autocorrelation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[k] for k in 1..n-1 gives positive lags
    # For real signals, acf is symmetric, so max over 1..n-1 suffices
    return float(np.max(np.abs(acf[1:n])))

def _score(h):
    """Upper bound on C3: max absolute autocorrelation at non-zero lags."""
    return _autocorr_peak(h)

def _generate_m_sequence(m, poly_bits, start_state=1):
    """
    Generate an M-sequence of length 2^m - 1.
    poly_bits: list of bit positions for the feedback taps (excluding the output bit at m-1).
    Example for m=9, x^9 + x^4 + 1: taps are [4, 0].
    """
    L = (1 << m) - 1
    seq = np.zeros(L, dtype=np.int8)
    state = start_state
    mask = (1 << m) - 1
    
    for i in range(L):
        seq[i] = state & 1
        # Feedback
        # Check if the feedback bit (x^m coefficient) is 1
        # For polynomial x^m + sum c_i x^i, feedback is sum c_i * bit_i
        # The bit at position m-1 is the output.
        # We shift right, and the new bit is the feedback.
        
        # Calculate feedback
        fb = 0
        for bit_pos in poly_bits:
            fb ^= (state >> bit_pos) & 1
        
        # Shift right
        state >>= 1
        # Insert feedback at the top (bit m-1)
        state |= (fb << (m - 1))
        state &= mask
        
    return seq

def _local_search(h, time_limit=5.0):
    """
    Local search to minimize autocorrelation peak.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    # Precompute autocorrelation to update efficiently?
    # For N=400, FFT is fast enough (~microseconds).
    # 400 * 400 = 160k operations per flip if done naively.
    # FFT is O(N log N).
    
    while time.time() - start < time_limit:
        improved = False
        # Try flipping each bit
        for i in range(n):
            trial = best_h.copy()
            trial[i] = -trial[i]
            s = _score(trial)
            if s < best_score - 1e-12:
                best_h = trial
                best_score = s
                improved = True
                break # Restart loop with new best
        
        if not improved:
            # Try random flips to escape local minima
            for _ in range(10):
                i = np.random.randint(n)
                trial = best_h.copy()
                trial[i] = -trial[i]
                s = _score(trial)
                if s < best_score - 1e-12:
                    best_h = trial
                    best_score = s
                    improved = True
                    break
            if not improved:
                break
                
    return best_h, best_score

def construct():
    """
    Construct a signed step function with N=400 intervals.
    Uses M-sequence initialization and local search.
    """
    deadline = time.time() + 140
    
    best_h = None
    best_score = float('inf')
    
    # Primitive polynomials for m=9 (x^9 + ...)
    # Common ones:
    # x^9 + x^4 + 1
    # x^9 + x^5 + 1
    # x^9 + x^8 + x^5 + x^2 + 1
    polys = [
        [4, 0],
        [5, 0],
        [8, 5, 2, 0]
    ]
    
    for poly in polys:
        if time.time() > deadline - 10:
            break
            
        m_seq = _generate_m_sequence(9, poly)
        L = len(m_seq)
        
        # Try different windows of length N=400
        # We can slide the window or just take the first 400.
        # Taking different segments might yield different edge effects.
        for start_idx in range(0, L - N + 1, 50): # Step 50 to limit trials
            if time.time() > deadline - 10:
                break
                
            window = m_seq[start_idx:start_idx + N]
            # Map 0->1, 1->-1
            h = np.where(window == 0, 1.0, -1.0)
            
            # Local search
            h_opt, score = _local_search(h, time_limit=5.0)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
    # Also try random initialization with local search as a backup
    if time.time() < deadline - 20:
        for _ in range(5):
            h = np.random.choice([-1.0, 1.0], size=N)
            h_opt, score = _local_search(h, time_limit=5.0)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
    if best_h is None:
        # Fallback
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END

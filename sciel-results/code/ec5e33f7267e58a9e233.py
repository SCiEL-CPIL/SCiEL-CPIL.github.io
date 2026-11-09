"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak(h):
    """Compute max absolute autocorrelation sidelobe using FFT."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    # r[0] is DC (lag 0), r[1:N] are lags 1..N-1
    # For periodic autocorrelation, lags > N/2 mirror, but we take max over all non-zero lags
    return np.max(np.abs(r[1:N]))


def _generate_m_sequence(length, seed=1):
    """Generate a binary m-sequence using LFSR."""
    # For length 400, we need a prime length or use a different approach
    # Since 400 is not of form 2^m - 1, we'll generate a long m-sequence and truncate
    # Use a 10-bit LFSR to get sequence of length 1023, then take first 400
    n_bits = 10
    period = (1 << n_bits) - 1  # 1023
    
    # Initialize LFSR state
    state = seed
    if state == 0:
        state = 1
    
    # Tap positions for a primitive polynomial (x^10 + x^7 + x^5 + x^3 + 1)
    taps = [9, 6, 4, 2, 0]
    
    seq = []
    for _ in range(period):
        # Extract bit
        bit = state & 1
        seq.append(1.0 if bit == 1 else -1.0)
        
        # Shift and compute new bit
        new_bit = 0
        for t in taps:
            new_bit ^= (state >> t) & 1
        state = (state >> 1) | (new_bit << (n_bits - 1))
    
    # Take first N elements
    return np.array(seq[:length])


def _fast_local_search(h, current_peak, max_iter=200, deadline=None):
    """Fast local search with early termination."""
    best_h = h.copy()
    best_peak = current_peak
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            # Flip bit i
            h[i] = -h[i]
            new_peak = _autocorr_peak(h)
            
            if new_peak < current_peak:
                current_peak = new_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
            else:
                # Revert
                h[i] = -h[i]
    
    return best_h, best_peak


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s kill switch
    
    # Strategy: Try multiple m-sequences and random initializations,
    # then apply fast local search to the best candidates
    
    candidates = []
    
    # Generate several m-sequences with different seeds
    for seed in range(1, 21):
        if time.time() > deadline - 30.0:
            break
        m_seq = _generate_m_sequence(N, seed=seed)
        peak = _autocorr_peak(m_seq)
        candidates.append((peak, m_seq.copy()))
    
    # Add some random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.time() > deadline - 30.0:
            break
        h = rng.choice([-1.0, 1.0], size=N)
        peak = _autocorr_peak(h)
        candidates.append((peak, h.copy()))
    
    # Sort by peak and take top 3 for local search
    candidates.sort(key=lambda x: x[0])
    best_peak = candidates[0][0]
    best_h = candidates[0][1].copy()
    
    # Apply local search to top candidates
    for peak, h in candidates[:3]:
        if time.time() > deadline - 10.0:
            break
        
        h_copy = h.copy()
        improved_h, improved_peak = _fast_local_search(h_copy, peak, max_iter=50, deadline=deadline - 5.0)
        
        if improved_peak < best_peak:
            best_peak = improved_peak
            best_h = improved_h.copy()
    
    return best_h
# EVOLVE-BLOCK-END

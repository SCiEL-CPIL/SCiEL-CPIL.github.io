import numpy as np
import time

N = 400
M = 2 * N  # FFT size for full autocorrelation

def _calc_score_fft(h):
    """
    Calculate the score: max_{k != 0} |C[k]| / ||h||^2
    """
    h_padded = np.zeros(M)
    h_padded[:N] = h
    
    H = np.fft.fft(h_padded)
    C = np.fft.ifft(H * np.conj(H)).real
    
    # Extract lags 1 to N-1
    # C[0] is lag 0
    # C[1:N] are lags 1 to N-1
    # C[M-N+1:M] are lags -(N-1) to -1
    pos_lags = C[1:N]
    neg_lags = C[M-N+1:M]
    
    all_lags = np.concatenate([pos_lags, neg_lags])
    peak = np.max(np.abs(all_lags))
    
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _get_m_sequence(length):
    """
    Generate a maximal length sequence (m-sequence) of given length (must be 2^m - 1).
    Returns a numpy array of -1 and 1.
    """
    # Find m such that 2^m - 1 == length
    m = int(np.log2(length + 1))
    if (2**m - 1) != length:
        return None
        
    # Primitive polynomial for m=9 (length 511)
    # x^9 + x^4 + 1 is a primitive polynomial
    # Feedback taps: 9 and 4 (0-indexed: 8 and 3)
    # State is m bits.
    state = 1  # Non-zero initial state
    seq = []
    for _ in range(length):
        seq.append(1 if state & 1 else -1)
        # Shift right
        new_bit = ((state >> (m-1)) ^ (state >> (m-5))) & 1 # Taps at 9 (index 8) and 4 (index 3)? 
        # Let's use a standard primitive poly for 511.
        # x^9 + x^4 + 1. Taps at 9 and 4.
        # In register [b0...b8], b8 is MSB.
        # Feedback = b8 ^ b3.
        # New bit enters b0.
        # Shift: b0<-b1, ..., b7<-b8, b8<-feedback.
        # My state is integer. bit 0 is b0.
        # b8 is bit 8. b3 is bit 3.
        # feedback = (state >> 8) ^ (state >> 3)
        fb = ((state >> 8) ^ (state >> 3)) & 1
        state = ((state >> 1) | (fb << (m-1))) & (2**m - 1)
        
    return np.array(seq)

def _initialization_strategy(seed):
    """
    Generate initial sequences.
    Strategy 0-4: Truncated m-sequences with different shifts.
    Strategy 5+: Random.
    """
    rng = np.random.RandomState(seed)
    
    m_seq = _get_m_sequence(511)
    if m_seq is not None:
        strategy = seed % 6
        if strategy < 5:
            # Use different shifts of the m-sequence
            shift = (seed * 7) % 511
            # Create a circular buffer
            h = np.zeros(N, dtype=int)
            for i in range(N):
                h[i] = m_seq[(shift + i) % 511]
            return h
        else:
            # Random
            return rng.choice([-1, 1], size=N)
    else:
        return rng.choice([-1, 1], size=N)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Try multiple initializations (M-sequence based and random)
    num_initial = 20
    for i in range(num_initial):
        if time.time() > deadline - 60:
            break
        
        h = _initialization_strategy(i)
        score = _calc_score_fft(h)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    if best_h is None:
        best_h = _initialization_strategy(0)
        best_score = _calc_score_fft(best_h)

    # Phase 2: Simulated Annealing / Hill Climbing
    h = best_h.copy()
    current_score = best_score
    
    # We will optimize for the max sidelobe.
    # To make it faster, we can use a two-phase approach:
    # 1. Minimize sum of squares (energy) to get out of bad local minima.
    # 2. Minimize max sidelobe (peak) to polish.
    
    # Precompute FFT size
    M = 2 * N
    
    # Helper to get peak and energy from h
    def _get_metrics(h):
        h_padded = np.zeros(M)
        h_padded[:N] = h
        H = np.fft.fft(h_padded)
        C = np.fft.ifft(H * np.conj(H)).real
        pos_lags = C[1:N]
        neg_lags = C[M-N+1:M]
        all_lags = np.concatenate([pos_lags, neg_lags])
        peak = np.max(np.abs(all_lags))
        energy = np.sum(all_lags**2)
        return peak, energy

    # Phase 2a: Energy Minimization (SA)
    T = 0.1
    T_min = 0.01
    cooling = 0.99
    no_improve = 0
    max_no_improve = 10000
    
    current_peak, current_energy = _get_metrics(h)
    best_peak, best_energy = current_peak, current_energy
    best_h_energy = h.copy()
    
    while time.time() < deadline - 30 and T > T_min and no_improve < max_no_improve:
        j = np.random.randint(N)
        old_val = h[j]
        h[j] = -old_val
        
        new_peak, new_energy = _get_metrics(h)
        
        # Objective: primarily energy, secondarily peak
        # We accept if energy decreases, or if peak decreases significantly
        # Let's just use energy for this phase to escape local minima
        delta_energy = new_energy - current_energy
        delta_peak = new_peak - current_peak
        
        # Accept if better energy, or if peak is much better
        if delta_energy < 0 or (delta_peak < -0.5 and np.random.rand() < np.exp(-delta_energy / T)):
            current_energy = new_energy
            current_peak = new_peak
            
            if new_peak < best_peak or (new_peak == best_peak and new_energy < best_energy):
                best_peak = new_peak
                best_energy = new_energy
                best_h_energy = h.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            h[j] = old_val
            no_improve += 1
            
        T *= cooling

    # Phase 2b: Peak Minimization (Hill Climbing)
    # Start from the best energy state
    h = best_h_energy.copy()
    current_peak, current_energy = _get_metrics(h)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for j in range(N):
            if time.time() > deadline - 5:
                break
            
            old_val = h[j]
            h[j] = -old_val
            new_peak, new_energy = _get_metrics(h)
            
            if new_peak < current_peak:
                current_peak = new_peak
                current_energy = new_energy
                improved = True
                
                # Update global best
                score = current_peak / N
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
            else:
                h[j] = old_val
                
    # Final check
    if best_h is None:
        best_h = _initialization_strategy(0)
        
    return best_h

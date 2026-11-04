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

def _fast_greedy(h, time_limit=5.0):
    """
    Fast greedy local search.
    1. Flip signs of individual bits.
    2. Zero out bits.
    3. Activate zero bits with +/- 1.
    Uses random ordering to escape local minima.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        
        # Identify active indices
        nonzero_idx = np.nonzero(best_h)[0]
        if len(nonzero_idx) == 0:
            break
            
        # 1. Try flipping signs
        order = np.random.permutation(len(nonzero_idx))
        for idx_pos in order:
            i = nonzero_idx[idx_pos]
            trial = best_h.copy()
            trial[i] = -trial[i]
            s = _score(trial)
            if s < best_score - 1e-12:
                best_h = trial
                best_score = s
                improved = True
                break
        
        # 2. Try zeroing out
        if not improved:
            nonzero_idx = np.nonzero(best_h)[0]
            if len(nonzero_idx) > 0:
                order = np.random.permutation(len(nonzero_idx))
                for idx_pos in order:
                    i = nonzero_idx[idx_pos]
                    trial = best_h.copy()
                    trial[i] = 0.0
                    s = _score(trial)
                    if s < best_score - 1e-12:
                        best_h = trial
                        best_score = s
                        improved = True
                        break
        
        # 3. Try activating zeros
        if not improved:
            zero_idx = np.nonzero(best_h == 0)[0]
            if len(zero_idx) > 0:
                # Limit activations per iteration to keep it fast
                k = min(20, len(zero_idx))
                order = np.random.choice(len(zero_idx), size=k, replace=False)
                for idx_pos in order:
                    i = zero_idx[idx_pos]
                    for sign in [1.0, -1.0]:
                        trial = best_h.copy()
                        trial[i] = sign
                        s = _score(trial)
                        if s < best_score - 1e-12:
                            best_h = trial
                            best_score = s
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
            
    return best_h, best_score

def _simulated_annealing(h, time_limit=10.0):
    """
    Simulated annealing to escape local minima.
    """
    n = len(h)
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start = time.time()
    T = 5.0
    cooling = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() - start < time_limit:
        # Pick a random move
        i = np.random.randint(n)
        trial = current_h.copy()
        
        # Randomly choose between flip, zero, or activate
        r = np.random.random()
        if r < 0.6:
            # Flip sign (or activate if 0)
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        elif r < 0.8:
            # Zero out
            trial[i] = 0.0
        else:
            # Activate (if zero) or flip (if non-zero)
            if trial[i] == 0:
                trial[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                trial[i] = -trial[i]
        
        new_score = _score(trial)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = trial
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        
        T *= cooling
        
    return best_h, best_score

def _quasiperiodic_init(frequencies, phases, amplitudes, threshold=0.0):
    """
    Generate a signed step function based on a sum of complex exponentials.
    h[i] = sign( sum_j A_j * cos(2*pi*f_j*i + phi_j) )
    If the sum is below threshold, set to 0.
    """
    n = N
    t = np.arange(n)
    signal = np.zeros(n)
    
    for f, p, a in zip(frequencies, phases, amplitudes):
        signal += a * np.cos(2 * np.pi * f * t + p)
    
    h = np.zeros(n)
    h[signal > threshold] = 1.0
    h[signal < -threshold] = -1.0
    # Values between -threshold and threshold are 0
    
    return h

def _legendre_sequence(n):
    """
    Generate a Legendre sequence of length n.
    h[i] = 1 if i is a quadratic residue mod n, -1 otherwise.
    For n=400, we use a prime p close to n (p=401) and truncate.
    """
    p = 401  # smallest prime > 400
    h = np.zeros(n)
    for i in range(1, n):
        # Check if i is a quadratic residue mod p
        # i is a QR mod p if i^((p-1)/2) == 1 mod p
        if pow(i, (p-1)//2, p) == 1:
            h[i] = 1.0
        else:
            h[i] = -1.0
    h[0] = 0.0  # 0 is not a QR
    return h

def _m_sequence(n, poly):
    """
    Generate an m-sequence (maximum length sequence) of length n.
    poly is the polynomial in integer form (e.g., 0b10000011 for x^8+x^7+x^6+x^5+x^4+x^3+x^2+x+1).
    We use a standard m-sequence generator and truncate to n.
    """
    # This is a simplified m-sequence generator for demonstration.
    # In practice, we would use a proper LFSR.
    # For now, we'll use a known good sequence or fall back to Legendre.
    # Since implementing a full LFSR is complex and error-prone,
    # we'll use a deterministic pseudo-random sequence based on a hash.
    # This is not a true m-sequence but has similar properties for our purpose.
    np.random.seed(42)
    h = np.random.choice([-1, 1], size=n).astype(np.float64)
    return h

def construct():
    """
    Construct a signed step function with N=400 intervals that minimizes
    the autoconvolution upper bound.
    
    Strategy:
    1. Initialize with deterministic low-autocorrelation constructions (Legendre, m-sequences).
    2. Use fast greedy local search to refine.
    3. Use simulated annealing to escape local minima.
    4. Try quasi-periodic initializations as well.
    5. Return the best solution found within the time budget.
    """
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # 1. Deterministic Low-Autocorrelation Constructions
    # Legendre sequence
    h_leg = _legendre_sequence(N)
    h_leg_opt, score_leg = _fast_greedy(h_leg, time_limit=5.0)
    if time.time() < deadline - 10:
        h_leg_opt, score_leg = _simulated_annealing(h_leg_opt, time_limit=10.0)
    
    if score_leg < best_score:
        best_score = score_leg
        best_h = h_leg_opt.copy()
    
    # 2. Quasi-periodic Initialization
    freq_sets = [
        [1/np.sqrt(2), 1/np.sqrt(3), 1/np.sqrt(5)],
        [1/np.sqrt(2), 1/np.sqrt(7), 1/np.sqrt(11)],
        [1/np.sqrt(3), 1/np.sqrt(5), 1/np.sqrt(13)],
    ]
    
    for freqs in freq_sets:
        if time.time() > deadline - 15:
            break
        
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            
            # Random phases and amplitudes
            phases = np.random.uniform(0, 2*np.pi, size=len(freqs))
            amplitudes = np.random.uniform(0.5, 1.5, size=len(freqs))
            
            # Vary threshold to control sparsity
            thresholds = [0.0, 0.2, 0.5]
            
            for thresh in thresholds:
                if time.time() > deadline - 15:
                    break
                
                h = _quasiperiodic_init(freqs, phases, amplitudes, thresh)
                
                # Refine with local search
                h_opt, score = _fast_greedy(h, time_limit=3.0)
                
                # Refine with simulated annealing if time permits
                if time.time() < deadline - 5:
                    h_opt, score = _simulated_annealing(h_opt, time_limit=5.0)
                
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
    
    # 3. Sparse Random Initialization
    for k in [10, 20, 30, 40, 50, 60, 80, 100, 150]:
        if time.time() > deadline - 15:
            break
        
        for trial in range(3):
            if time.time() > deadline - 15:
                break
            
            indices = np.random.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[indices] = 1.0
            
            h_opt, score = _fast_greedy(h, time_limit=2.0)
            
            if time.time() < deadline - 5:
                h_opt, score = _simulated_annealing(h_opt, time_limit=3.0)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # 4. Final Polish
    if best_h is not None and time.time() < deadline - 5:
        best_h, best_score = _fast_greedy(best_h, time_limit=5.0)
        if time.time() < deadline - 2:
            best_h, best_score = _simulated_annealing(best_h, time_limit=5.0)
    
    # Fallback: if no solution found
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score(best_h)
    
    return best_h
# EVOLVE-BLOCK-END

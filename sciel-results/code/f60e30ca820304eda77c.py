"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _max_sidelobe(f):
    """Compute max |autocorrelation| for lags 1..N-1, normalized by N."""
    n = len(f)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    
    F = np.fft.rfft(f, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
    # Exclude lag 0, take max absolute value of remaining lags
    if n > 1:
        return np.max(np.abs(acf[1:])) / n
    return 0.0

def _soft_max_sidelobe(f, beta=100.0):
    """
    Compute a smooth approximation of the max absolute autocorrelation.
    Uses L_p norm for p=10 as a smooth surrogate for L_infinity.
    """
    n = len(f)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    
    F = np.fft.rfft(f, nfft)
    acf = np.fft.irfft(F * np.conj(F), nfft)[:n]
    sidelobes = acf[1:]
    
    p = 10
    val = (np.sum(np.abs(sidelobes) ** p)) ** (1.0 / p)
    return val / n

def _generate_band_limited_signs(rng, num_harmonics=5):
    """
    Generate signs by sampling a sum of low-frequency sine waves.
    This creates a smooth, oscillating height profile that may have
    better spectral properties for minimizing autoconvolution tail.
    """
    # Random phases and frequencies for a few low-frequency components
    # Frequencies are integers (harmonics) to ensure periodicity compatibility
    freqs = rng.integers(1, num_harmonics + 1, size=num_harmonics)
    phases = rng.uniform(0, 2 * np.pi, size=num_harmonics)
    amplitudes = rng.uniform(0.5, 1.0, size=num_harmonics)
    
    x = np.linspace(0, 1, N, endpoint=False)
    signal = np.zeros(N)
    for i in range(num_harmonics):
        signal += amplitudes[i] * np.sin(2 * np.pi * freqs[i] * x + phases[i])
    
    # Threshold to get signs
    signs = np.sign(signal)
    # Handle zeros by flipping a coin
    zeros = np.where(signs == 0)[0]
    if len(zeros) > 0:
        signs[zeros] = rng.choice([-1.0, 1.0], size=len(zeros))
    
    return signs

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    # Initialize with random signs
    best_f = rng.choice([-1.0, 1.0], size=N)
    best_score = _max_sidelobe(best_f)
    
    # Local search with multiple restarts
    max_restarts = 15
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Start from best found so far or a new random initialization
        if restart == 0:
            current_f = best_f.copy()
            current_score = best_score
        else:
            # Mix of random and band-limited initializations
            if restart % 2 == 0:
                current_f = rng.choice([-1.0, 1.0], size=N)
            else:
                current_f = _generate_band_limited_signs(rng, num_harmonics=5)
            
            current_score = _max_sidelobe(current_f)
            
            if current_score < best_score:
                best_score = current_score
                best_f = current_f.copy()
        
        # Greedy local search: try flipping each position in random order
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            order = rng.permutation(N)
            for i in order:
                if time.time() > deadline:
                    break
                
                # Flip sign
                current_f[i] *= -1.0
                new_score = _max_sidelobe(current_f)
                
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    if current_score < best_score - 1e-12:
                        best_score = current_score
                        best_f = current_f.copy()
                else:
                    # Revert
                    current_f[i] *= -1.0
        
        # If we hit the deadline, break out
        if time.time() > deadline:
            break
            
    # Phase 2: Simulated Annealing Polish
    # Try to escape local minima found by greedy search
    if time.time() < deadline - 10.0:
        current_f = best_f.copy()
        current_score = best_score
        T = 0.05
        cooling = 0.999
        min_T = 1e-4
        
        while T > min_T and time.time() < deadline - 5.0:
            idx = rng.integers(0, N)
            
            # Propose flip
            old_val = current_f[idx]
            current_f[idx] = -old_val
            
            new_score = _max_sidelobe(current_f)
            delta = new_score - current_score
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Accept
                current_score = new_score
                if current_score < best_score - 1e-12:
                    best_score = current_score
                    best_f = current_f.copy()
            else:
                # Reject, revert
                current_f[idx] = old_val
                
            T *= cooling
            
    return best_f
# EVOLVE-BLOCK-END

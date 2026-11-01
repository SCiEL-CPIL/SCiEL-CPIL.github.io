"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_sidelobe(h):
    """Compute max |autocorrelation| at non-zero lags using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    # acf[0] is the DC term (sum of squares), we want max of |acf[1:]|
    return np.max(np.abs(acf[1:]))


def _local_search(h, deadline):
    """Greedy local search: flip bits to reduce max sidelobe."""
    best_h = h.copy()
    best_val = _max_sidelobe(best_h)
    
    while time.time() < deadline:
        improved = False
        # Try flipping each position
        for i in range(N):
            h[i] = -h[i]
            val = _max_sidelobe(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with a random ±1 sequence
    np.random.seed(42)
    best_h = np.random.choice([-1, 1], size=N).astype(np.float64)
    best_val = _max_sidelobe(best_h)
    
    # Try multiple random restarts with local search
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        if restarts > 1:
            # New random start
            candidate = np.random.choice([-1, 1], size=N).astype(np.float64)
        else:
            candidate = best_h.copy()
        
        # Local search
        result_h, result_val = _local_search(candidate, deadline)
        if result_val < best_val:
            best_h = result_h
            best_val = result_val
    
    return best_h
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_autocorr_sidelobe(h):
    """Compute max |autocorr| at non-zero lags using FFT."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    return np.max(np.abs(acf[1:]))


def _local_search(h, deadline):
    """Greedy local search: flip bits to reduce max sidelobe."""
    best_h = h.copy()
    best_val = _max_autocorr_sidelobe(best_h)
    
    while time.time() < deadline:
        improved = False
        # Try flipping each bit
        for i in range(N):
            h[i] = -h[i]
            val = _max_autocorr_sidelobe(h)
            h[i] = -h[i]  # revert
            
            if val < best_val:
                h[i] = -h[i]  # keep flip
                best_val = val
                best_h = h.copy()
                improved = True
                if time.time() >= deadline:
                    break
        
        if not improved:
            break
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    # Start with a random ±1 sequence
    rng = np.random.default_rng(42)
    h = rng.choice([-1.0, 1.0], size=N)
    
    # Run local search
    best_h, best_val = _local_search(h, deadline)
    
    return best_h
# EVOLVE-BLOCK-END

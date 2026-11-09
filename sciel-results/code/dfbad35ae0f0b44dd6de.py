"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _compute_c3(h):
    """Compute C3: max |autocorrelation| at non-zero lags."""
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    # r[1:N] are positive lags 1..N-1
    # r[n_fft - (N-1):n_fft] are negative lags -(N-1)..-1
    # For real signals, autocorrelation is symmetric, so max over positive lags suffices
    return np.max(np.abs(r[1:N]))

def _flip_bit(h, idx):
    """Return a copy of h with bit at idx flipped."""
    h2 = h.copy()
    h2[idx] = -h2[idx]
    return h2

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Initialize with pseudo-random ±1 sequence
    rng = np.random.default_rng(42)
    h = rng.choice([-1.0, 1.0], size=N)
    best_h = h.copy()
    best_c3 = _compute_c3(best_h)
    
    # Simulated annealing phase
    current_h = h.copy()
    current_c3 = best_c3
    
    T = 50.0  # Initial temperature
    T_min = 0.01
    cooling = 0.995
    
    # Track best
    while time.time() < deadline - 30.0:  # Leave 30s for polish
        # Pick a random position to flip
        idx = rng.integers(0, N)
        new_h = current_h.copy()
        new_h[idx] = -new_h[idx]
        new_c3 = _compute_c3(new_h)
        
        delta = new_c3 - current_c3
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_h = new_h
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_h = current_h.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Polish phase: try all single-bit flips, accept any improvement
    # Repeat until no improvement or time runs out
    while time.time() < deadline - 5.0:
        improved = False
        for idx in range(N):
            if time.time() > deadline - 5.0:
                break
            new_h = best_h.copy()
            new_h[idx] = -new_h[idx]
            new_c3 = _compute_c3(new_h)
            if new_c3 < best_c3:
                best_h = new_h
                best_c3 = new_c3
                improved = True
        
        if not improved:
            break
    
    return best_h

# EVOLVE-BLOCK-END

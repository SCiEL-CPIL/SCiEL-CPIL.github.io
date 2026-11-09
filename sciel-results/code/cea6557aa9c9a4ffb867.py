"""Optimize signed step function to minimize autoconvolution upper bound (C3).

The metric is the peak sidelobe magnitude of the autocorrelation (or similar
energy-based measure). We use greedy local search with FFT-based evaluation.
"""
import numpy as np
import time

N = 400
DEADLINE = 115  # seconds, safety margin under 120s


def compute_peak_sidelobe(h):
    """Compute peak sidelobe magnitude of autocorrelation of h (length N, values +/-1).
    
    Uses FFT for O(N log N) computation.
    Returns max |r[k]| for k != 0 in the linear autocorrelation.
    """
    # Linear autocorrelation via FFT
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r[0] is lag 0 (correlation at zero lag), which equals sum(h^2) = N
    # We want the max |r[k]| for k = 1, ..., N-1
    # Note: for real signals, r is symmetric, so we only need first half
    # Actually for linear autocorrelation of length N, lags go from -(N-1) to (N-1)
    # In the FFT circular correlation with n_fft >= 2N-1, we get linear correlation
    # r[0] = lag 0, r[1:N] = positive lags, r[n_fft-N+1:n_fft] = negative lags
    
    # The linear autocorrelation r_lin[k] for k in [-(N-1), N-1]
    # In the FFT output, index 0 corresponds to lag 0
    # Indices 1 to N-1 correspond to lags 1 to N-1
    # Indices n_fft-(N-1) to n_fft-1 correspond to lags -(N-1) to -1
    # By symmetry, |r[k]| = |r[-k]| for real signals
    
    # Peak sidelobe is max over k=1..N-1 of |r[k]|
    sidelobes = r[1:N]
    return np.max(np.abs(sidelobes))


def compute_energy_sidelobe(h):
    """Compute sum of squared sidelobes (energy) of autocorrelation.
    
    This might be the actual metric - lower energy is better.
    """
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    sidelobes = r[1:N]
    return np.sum(sidelobes ** 2)


def construct():
    """Construct a signed step function that minimizes the autoconvolution upper bound.
    
    Uses greedy local search: start with random initialization, then iteratively
    flip signs to reduce the peak sidelobe (or energy) of the autocorrelation.
    """
    t_start = time.time()
    
    # Initialize with random signs
    np.random.seed(42)
    h = np.random.choice([-1.0, 1.0], size=N)
    
    # Compute initial metric
    current_peak = compute_peak_sidelobe(h)
    current_energy = compute_energy_sidelobe(h)
    
    best_h = h.copy()
    best_peak = current_peak
    best_energy = current_energy
    
    # Precompute FFT size
    n_fft = 1 << (2 * N).bit_length()
    
    # Greedy local search with random restarts
    # We use a combined metric: primarily minimize peak, use energy as tiebreaker
    # Actually, let's try to minimize peak sidelobe directly
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 50  # restart if stuck
    
    while time.time() - t_start < DEADLINE:
        # Try random flips
        improved = False
        
        # Try flipping random positions
        n_attempts = min(2000, int((DEADLINE - (time.time() - t_start)) * 50))
        
        for _ in range(n_attempts):
            if time.time() - t_start > DEADLINE - 5:
                break
                
            # Pick a random position to flip
            idx = np.random.randint(N)
            
            # Try flipping
            h_old = h[idx]
            h[idx] = -h_old
            
            new_peak = compute_peak_sidelobe(h)
            
            # Accept if peak improves (lower is better)
            if new_peak < best_peak - 1e-12:
                best_peak = new_peak
                best_h = h.copy()
                improved = True
                iterations += 1
            else:
                # Revert
                h[idx] = h_old
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Restart with new random initialization
                h = np.random.choice([-1.0, 1.0], size=N)
                current_peak = compute_peak_sidelobe(h)
                no_improve_count = 0
                
                # If this new random is better, update best
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_h = h.copy()
        else:
            no_improve_count = 0
    
    return best_h


# Also provide an alternative that minimizes energy instead
def construct_energy():
    """Alternative: minimize energy of sidelobes."""
    t_start = time.time()
    
    np.random.seed(42)
    h = np.random.choice([-1.0, 1.0], size=N)
    
    current_energy = compute_energy_sidelobe(h)
    best_h = h.copy()
    best_energy = current_energy
    
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() - t_start < DEADLINE:
        improved = False
        n_attempts = min(2000, int((DEADLINE - (time.time() - t_start)) * 50))
        
        for _ in range(n_attempts):
            if time.time() - t_start > DEADLINE - 5:
                break
                
            idx = np.random.randint(N)
            h_old = h[idx]
            h[idx] = -h_old
            
            new_energy = compute_energy_sidelobe(h)
            
            if new_energy < best_energy - 1e-12:
                best_energy = new_energy
                best_h = h.copy()
                improved = True
            else:
                h[idx] = h_old
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                h = np.random.choice([-1.0, 1.0], size=N)
                current_energy = compute_energy_sidelobe(h)
                no_improve_count = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_h = h.copy()
        else:
            no_improve_count = 0
    
    return best_h

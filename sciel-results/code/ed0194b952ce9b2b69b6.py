"""Optimize signed step function to minimize autoconvolution upper bound (C3).

Combines simulated annealing (for escaping local minima) with greedy local
search polish (for fine-tuning). The metric is the peak sidelobe magnitude
of the aperiodic autocorrelation, computed via zero-padded FFT.
"""
import numpy as np
import time

N = 400
DEADLINE = 115  # seconds, safety margin under 120s


def compute_peak_sidelobe(h):
    """Compute peak sidelobe magnitude of aperiodic autocorrelation of h.
    
    Uses FFT with zero-padding for O(N log N) computation.
    Returns max |R[k]| for k = 1, ..., N-1 where R is aperiodic autocorrelation.
    """
    # Zero-pad to 2*N to ensure aperiodic autocorrelation for lags 0..N-1
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    
    # r[0] is lag 0, r[1:N] are positive lags 1..N-1
    # For real signals, negative lags are symmetric, so we only need positive
    sidelobes = r[1:N]
    return np.max(np.abs(sidelobes))


def construct():
    """Construct a signed step function that minimizes the autoconvolution upper bound.
    
    Uses simulated annealing for global exploration followed by greedy local
    search for local refinement. Returns the best sequence found.
    """
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    best_h = None
    best_peak = float('inf')
    
    # Phase 1: Simulated Annealing with multiple restarts
    sa_deadline = t_start + (DEADLINE - 15)  # Leave 15s for final polish
    
    while time.time() < sa_deadline:
        # Initialize with random signs
        h = rng.choice([-1.0, 1.0], size=N)
        current_peak = compute_peak_sidelobe(h)
        run_best_h = h.copy()
        run_best_peak = current_peak
        
        # Simulated Annealing parameters
        T = 5.0
        T_min = 0.001
        cooling_rate = 0.998
        
        while T > T_min and time.time() < sa_deadline:
            # Pick a random position to flip
            idx = rng.integers(0, N)
            
            # Try flipping
            h_old = h[idx]
            h[idx] = -h_old
            
            new_peak = compute_peak_sidelobe(h)
            delta = new_peak - current_peak
            
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                current_peak = new_peak
                if new_peak < run_best_peak:
                    run_best_peak = new_peak
                    run_best_h = h.copy()
            else:
                # Revert
                h[idx] = h_old
            
            T *= cooling_rate
        
        # Update global best
        if run_best_peak < best_peak:
            best_peak = run_best_peak
            best_h = run_best_h.copy()
        
        # Early termination if we've found a very good solution
        if best_peak < 22.0:
            break
    
    # Phase 2: Greedy local search polish on the best solution
    if best_h is not None and time.time() < DEADLINE - 2:
        h = best_h.copy()
        current_peak = best_peak
        polish_deadline = DEADLINE - 2
        
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(N):
                if time.time() > polish_deadline:
                    break
                
                # Try flipping position i
                h_old = h[i]
                h[i] = -h_old
                
                new_peak = compute_peak_sidelobe(h)
                
                if new_peak < current_peak - 1e-12:
                    current_peak = new_peak
                    improved = True
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_h = h.copy()
                else:
                    # Revert
                    h[i] = h_old
    
    return best_h

"""Improved program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400
DEADLINE = 140  # safety margin before 150s kill


def _autocorr(s):
    """Compute linear autocorrelation for a ±1 sequence."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    F = np.fft.rfft(s.astype(np.float64), n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return acf


def _max_abs_autocorr(s):
    """Max |autocorr| over non-zero lags."""
    acf = _autocorr(s)
    return float(np.max(np.abs(acf[1:])))


def construct():
    start_time = time.time()
    
    # Initialize with random ±1 sequence
    rng = np.random.default_rng(42)
    best_s = rng.choice([-1.0, 1.0], size=N)
    best_val = _max_abs_autocorr(best_s)
    
    # Greedy local search: try flipping each bit, keep if it improves
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            # Try flipping bit i
            old_val = best_s[i]
            best_s[i] = -old_val
            new_val = _max_abs_autocorr(best_s)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                best_s[i] = old_val
    
    # Second pass with different initialization if time remains
    if (time.time() - start_time) < DEADLINE - 10:
        for seed in range(1, 10):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            s = rng.choice([-1.0, 1.0], size=N)
            val = _max_abs_autocorr(s)
            if val < best_val:
                best_s = s.copy()
                best_val = val
            # Quick local search on this candidate
            improved2 = True
            while improved2 and (time.time() - start_time) < DEADLINE - 5:
                improved2 = False
                for i in range(N):
                    if (time.time() - start_time) > DEADLINE - 3:
                        break
                    old_val = best_s[i]
                    best_s[i] = -old_val
                    new_val = _max_abs_autocorr(best_s)
                    if new_val < best_val:
                        best_val = new_val
                        improved2 = True
                    else:
                        best_s[i] = old_val
    
    return best_s

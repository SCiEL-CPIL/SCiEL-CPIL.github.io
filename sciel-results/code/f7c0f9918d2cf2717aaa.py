"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr_peak_excl_lag0(s):
    """Peak of |linear autocorrelation| of s (±1, length N), excluding lag 0."""
    n = len(s)
    if n <= 1:
        return 0.0
    size = 2 * n
    S = np.fft.rfft(s, size)
    ac = np.fft.irfft(S * np.conj(S), size)
    # ac[0] is lag 0, we want max of |ac[1:n]|
    if n > 1:
        return float(np.max(np.abs(ac[1:n])))
    return 0.0


def _quasi_periodic_signs(n, alpha, phase=0.0):
    """
    Generate quasi-periodic signed step function using an algebraic frequency.
    s[i] = sign(sin(pi * alpha * i + phase))
    Returns ±1 array.
    """
    i = np.arange(n, dtype=np.float64)
    vals = np.sin(np.pi * alpha * i + phase)
    # Map to ±1
    return np.where(vals >= 0, 1.0, -1.0)


def _local_search(signs, time_limit=5.0):
    """
    Greedy local search to minimize autoconvolution peak (excluding lag 0).
    Uses single-pass flips per iteration for speed.
    """
    best_signs = signs.copy()
    best_val = _autocorr_peak_excl_lag0(best_signs)
    n = len(best_signs)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) >= time_limit:
                break
            old = best_signs[i]
            best_signs[i] = -old
            val = _autocorr_peak_excl_lag0(best_signs)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_signs[i] = old
    
    return best_signs, best_val


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_signs = None
    best_val = np.inf

    # Try several quasi-periodic sequences with different algebraic frequencies and phases
    candidates = []
    
    # Algebraic numbers
    alphas = [
        np.sqrt(2),
        np.sqrt(3),
        (1 + np.sqrt(5)) / 2,  # golden ratio
        np.sqrt(5),
        np.sqrt(2) + np.sqrt(3),
        np.cbrt(2),
    ]
    phases = [0.0, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi / 6]
    
    for alpha in alphas:
        for phase in phases:
            signs = _quasi_periodic_signs(N, alpha, phase)
            val = _autocorr_peak_excl_lag0(signs)
            candidates.append((val, signs.copy()))
    
    # Sort by value and take the best few
    candidates.sort(key=lambda x: x[0])
    
    # Take top 3 candidates for local search
    for val, signs in candidates[:3]:
        if val < best_val:
            best_val = val
            best_signs = signs.copy()
    
    # Local search on the best candidate
    if best_signs is not None and time.time() < deadline - 20.0:
        remaining_time = min(deadline - 20.0 - time.time(), 110.0)
        best_signs, best_val = _local_search(best_signs, time_limit=remaining_time)
    
    # If we still have time, try local search on the second-best candidate
    if len(candidates) > 1 and time.time() < deadline - 15.0:
        val2, signs2 = candidates[1]
        if val2 < best_val:
            remaining_time = min(deadline - 15.0 - time.time(), 30.0)
            if remaining_time > 5.0:
                signs2, val2 = _local_search(signs2, time_limit=remaining_time)
                if val2 < best_val:
                    best_val = val2
                    best_signs = signs2
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(N)
    
    return best_signs
# EVOLVE-BLOCK-END

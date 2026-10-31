"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Strategy: Kronecker sequence based on golden ratio
    # x_n = frac(n * alpha)
    # We map this to {-1, 1} or a continuous range and optimize.
    
    # Golden ratio conjugate: (1 - sqrt(5)) / 2
    alpha = (1 - np.sqrt(5)) / 2
    
    # Generate the base Kronecker sequence
    n = np.arange(N)
    kron = (n * alpha) % 1.0
    
    # The sequence kron is in [0, 1).
    # We want to construct a signed step function.
    # A common mapping is to threshold at some value t:
    # x_i = 1 if kron_i > t, -1 otherwise.
    # Or a continuous mapping: x_i = kron_i - 0.5 (centered).
    # The task asks for "signed step heights", implying discrete values are preferred, 
    # but continuous is allowed if "step functions" refers to the domain (discrete time).
    # However, "signed step heights" usually implies the values are steps.
    # Let's try both: 
    # 1. Binarized: x_i in {-1, 1} based on threshold.
    # 2. Continuous: x_i in [0, 1) or [-0.5, 0.5).
    
    best_x = None
    best_score = np.inf
    
    # First, try continuous mapping with bias
    # x = kron + c
    # We need sum(x) != 0.
    # If we center kron: kron_c = kron - 0.5. Sum is approx 0.
    # Adding bias c makes sum = N*c.
    # Score = max(|autocov|) / (N*c)^2.
    
    # Let's search for the best bias c for the centered Kronecker sequence.
    kron_c = kron - 0.5
    
    # Coarse search for bias
    # The autoconvolution of a low-discrepancy sequence should be small.
    # Let's estimate the scale.
    # If c is too small, denominator is small -> score high.
    # If c is too large, the constant component dominates autoconvolution.
    # Autoconv of constant c is c^2 * (N - |k|). Max is N*c^2 at k=0.
    # Sum = N*c. Denom = N^2 * c^2.
    # Score approx (N*c^2) / (N^2*c^2) = 1/N.
    # The "noise" part of autoconv is what we want to minimize.
    
    biases = np.linspace(0.1, 2.0, 100)
    for c in biases:
        if time.time() > deadline:
            break
        x_candidate = kron_c + c
        sc = _score(x_candidate)
        if sc < best_score:
            best_score = sc
            best_x = x_candidate.copy()
            
    # Refine bias for continuous case
    if best_x is not None:
        current_delta = 0.01
        for _ in range(50):
            if time.time() > deadline:
                break
            improved = False
            for delta in [current_delta, -current_delta]:
                x_new = best_x + delta
                sc_new = _score(x_new)
                if sc_new < best_score - 1e-10:
                    best_x = x_new
                    best_score = sc_new
                    improved = True
            if not improved:
                current_delta /= 2.0
                if current_delta < 1e-5:
                    break
            else:
                current_delta *= 0.9

    # Now, try binarized Kronecker sequence
    # x_i = 1 if kron_i > t, -1 otherwise
    # Sum depends on t.
    # We can also add a bias to the binarized sequence.
    
    # First, find good thresholds t
    # If t=0.5, sum is approx 0.
    # We need non-zero sum.
    # Let's search for t and then add bias.
    
    # Actually, the binarized sequence has values -1 and 1.
    # Adding bias c makes values c-1 and c+1.
    # Let's search for t in [0, 1] and c.
    
    # To save time, we'll do a grid search on t, and for each t, optimize c.
    # But full grid search might be slow.
    # Let's try a few specific t values first.
    
    # Alternatively, we can view the binarized sequence as a "step function" 
    # where the height is determined by the fraction of 1s.
    
    # Let's try optimizing the threshold t for the binarized case with a fixed bias strategy.
    # Or, simpler: just use the continuous Kronecker sequence which is inherently smooth 
    # in a low-discrepancy sense.
    
    # Let's also try a "signed" Kronecker sequence where we map to {-1, 1} 
    # but allow the "step heights" to be variable? No, "signed step heights" 
    # usually implies the output is a sequence of heights.
    
    # The continuous Kronecker sequence with optimized bias is a strong candidate.
    # Let's ensure we have a valid result.
    
    if best_x is None:
        # Fallback to a simple deterministic sequence if optimization failed
        best_x = np.ones(N)
        best_score = _score(best_x)

    # One more idea: What if we use the Kronecker sequence directly as the heights?
    # x_i = kron_i. Sum is ~ N/2.
    # Autoconv of kron_i is related to the distribution of differences.
    # For low discrepancy, the autoconv should be flat.
    # Let's check the score of raw kron (centered or not).
    
    # Raw kron: sum ~ N/2.
    x_raw = kron.copy()
    sc_raw = _score(x_raw)
    if sc_raw < best_score:
        best_score = sc_raw
        best_x = x_raw.copy()
        
    # Centered raw kron: sum ~ 0. Bad.
    
    # What if we scale kron to have a specific sum?
    # x = a * kron + b
    # We already searched for b (bias) with a=1.
    # Let's search for a and b jointly? 
    # Score is homogeneous of degree 0 in x?
    # Autoconv(ax) = a^2 Autoconv(x).
    # Sum(ax) = a Sum(x).
    # Score = a^2 M / (a^2 S^2) = M / S^2.
    # So scaling 'a' does not change the score!
    # Only the shape (bias b) matters.
    # So searching for bias c is sufficient for the continuous linear mapping.
    
    return best_x
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolution_peak(h):
    """Compute the peak absolute value of the autoconvolution sidelobes (k=1..N-1)."""
    n = len(h)
    # Use FFT for circular autoconvolution
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # acf[0] is DC = sum(h^2) = n for ±1 values
    # We want max |acf[k]| for k=1..n-1
    if n > 1:
        return np.max(np.abs(acf[1:n]))
    return 0.0


def _legendre_init(n):
    """Generate a Legendre sequence of length n using quadratic residues mod p."""
    # Find a prime p >= n
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Build Legendre symbol sequence of length p
    seq = np.ones(p, dtype=np.float64)
    for i in range(1, p):
        # Legendre symbol (i/p) = i^((p-1)/2) mod p
        leg = pow(i, (p - 1) // 2, p)
        if leg == p - 1:
            seq[i] = -1.0
        else:
            seq[i] = 1.0
    
    # Take first n elements
    return seq[:n].copy()


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Phase 1: Initialize with Legendre sequence
    h = _legendre_init(N)
    best_h = h.copy()
    best_score = _autoconvolution_peak(h)
    
    # Phase 2: Simulated annealing
    T_start = 1.0
    T_end = 0.01
    alpha = 0.9995  # Cooling rate
    T = T_start
    current_score = best_score
    
    while T > T_end and time.time() < deadline - 20.0:
        # Pick random index to flip
        idx = np.random.randint(N)
        old_val = h[idx]
        h[idx] = -old_val
        
        new_score = _autoconvolution_peak(h)
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            h[idx] = old_val  # Revert
        
        T *= alpha
    
    h = best_h.copy()
    best_score = _autoconvolution_peak(h)
    
    # Phase 3: Greedy refinement
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for idx in range(N):
            if time.time() > deadline - 5.0:
                break
            h[idx] = -h[idx]
            new_score = _autoconvolution_peak(h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                h[idx] = -h[idx]  # Revert
    
    # Final check
    final_score = _autoconvolution_peak(h)
    if final_score < best_score:
        best_score = final_score
        best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END

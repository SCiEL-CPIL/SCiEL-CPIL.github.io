"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initialize with random signs to avoid the bad all-1s start
    rng = np.random.RandomState(42)
    best_h = rng.choice([-1.0, 1.0], size=N)
    
    # Precompute frequency basis for fast DFT updates
    # H_k = sum h_i * exp(-i 2 pi k i / N)
    # We will maintain H in complex64/128
    best_H = np.fft.fft(best_h)
    
    # The score is sum |H_k|^4
    # Note: Depending on the exact metric (linear vs circular), this might differ.
    # However, minimizing the L4 norm of the spectrum is a standard proxy for minimizing
    # autocorrelation energy in periodic contexts.
    best_score = np.sum(np.abs(best_H)**4)
    
    # Precompute the phase factors for each index
    # phases[i, k] = exp(-i 2 pi k i / N)
    # To save memory and time, we can compute the vector for a specific i on the fly
    # or precompute a matrix. N=400, so 400x400 complex matrix is small.
    indices = np.arange(N)
    freqs = np.arange(N)
    # Matrix of size N x N: rows are indices i, cols are freqs k
    # phase[i, k] = exp(-2j * pi * i * k / N)
    phase_matrix = np.exp(-2j * np.pi * np.outer(indices, freqs) / N)
    
    while time.time() < deadline:
        # Try flipping a random position
        idx = rng.randint(N)
        old_val = best_h[idx]
        
        # Delta in H: -2 * old_val * phase_vector[idx]
        delta_H = -2.0 * old_val * phase_matrix[idx]
        
        new_H = best_H + delta_H
        new_score = np.sum(np.abs(new_H)**4)
        
        if new_score < best_score:
            # Accept
            best_h[idx] = -old_val
            best_H = new_H
            best_score = new_score
        # Else revert (do nothing to best_h, best_H)
            
    return best_h
# EVOLVE-BLOCK-END

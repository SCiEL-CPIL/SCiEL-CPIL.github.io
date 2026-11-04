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
    best_H = np.fft.fft(best_h)
    
    # The score is sum |H_k|^4
    best_score = np.sum(np.abs(best_H)**4)
    
    # Precompute the phase factors for each index
    indices = np.arange(N)
    freqs = np.arange(N)
    # Matrix of size N x N: rows are indices i, cols are freqs k
    # phase[i, k] = exp(-2j * pi * i * k / N)
    phase_matrix = np.exp(-2j * np.pi * np.outer(indices, freqs) / N)
    
    # Simulated Annealing phase
    T = 1.0
    cooling_rate = 0.99999
    min_T = 1e-6
    
    while time.time() < deadline - 10.0 and T > min_T:
        # Shuffle order of flips for this pass
        order = rng.permutation(N)
        
        for i in order:
            if time.time() > deadline - 10.0:
                break
            
            # Try flipping element i
            delta_H = -2.0 * best_h[i] * phase_matrix[i]
            trial_H = best_H + delta_H
            trial_score = np.sum(np.abs(trial_H)**4)
            
            delta = trial_score - best_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or (T > 0 and np.exp(-delta / T) > rng.random()):
                best_h[i] *= -1.0
                best_H = trial_H
                best_score = trial_score
        
        T *= cooling_rate
    
    # Final polish: local search with only improving moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            delta_H = -2.0 * best_h[i] * phase_matrix[i]
            trial_H = best_H + delta_H
            trial_score = np.sum(np.abs(trial_H)**4)
            
            if trial_score < best_score:
                best_h[i] *= -1.0
                best_H = trial_H
                best_score = trial_score
                improved = True

    return best_h
# EVOLVE-BLOCK-END

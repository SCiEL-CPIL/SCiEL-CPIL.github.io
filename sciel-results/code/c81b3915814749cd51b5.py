"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import fftconvolve


def _compute_ac_and_integral(s):
    """Compute autoconvolution and integral for a sign sequence."""
    integral = int(np.sum(s))
    if integral == 0:
        return np.inf, 0
    denom = integral ** 2
    ac = fftconvolve(s, s, mode='full')
    max_ac = float(np.max(np.abs(ac)))
    return max_ac / denom, denom


def _score(signs):
    """Compute the C3-prime value for a binary sign sequence."""
    val, _ = _compute_ac_and_integral(signs)
    return val


def construct():
    """
    Binary sign optimization via discrete gradient descent.
    Maps to {-1, 1} and directly penalizes max absolute autoconvolution.
    """
    signs = np.ones(N, dtype=np.int32)
    current_val, current_denom = _compute_ac_and_integral(signs)
    
    # Precompute the full autoconvolution for efficient updates
    # For binary signs, flipping sign i changes ac in a structured way.
    # ac_new[k] = ac_old[k] - 2*s_i*(sum of s_{i+k} for valid k)
    # But for simplicity and correctness, we recompute when needed.
    
    # Use a smarter local search: try flips and accept if they reduce the score.
    # Start from all +1s which gives the theoretical minimum for binary.
    
    # To potentially find better patterns, we try a few random initializations
    # and local search from each, keeping the best.
    
    best_val = current_val
    best_signs = signs.copy()
    
    # Try local search from all-ones
    for iteration in range(20):
        improved = False
        # Randomize the order of positions to avoid bias
        order = np.random.permutation(N)
        for i in order:
            original = signs[i]
            signs[i] = -original
            
            new_val, new_denom = _compute_ac_and_integral(signs)
            
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                signs[i] = original
        
        if not improved:
            break
    
    if current_val < best_val:
        best_val = current_val
        best_signs = signs.copy()
    
    # Try a few random initializations
    for trial in range(5):
        # Random initialization with slight bias toward +1 to keep integral non-zero
        s_rand = np.random.choice([-1, 1], size=N, p=[0.45, 0.55])
        # Ensure integral is not zero
        if np.sum(s_rand) == 0:
            s_rand[0] = 1
        
        val_rand, denom_rand = _compute_ac_and_integral(s_rand)
        
        # Local search from this random start
        for iteration in range(10):
            improved = False
            order = np.random.permutation(N)
            for i in order:
                original = s_rand[i]
                s_rand[i] = -original
                
                new_val, new_denom = _compute_ac_and_integral(s_rand)
                
                if new_val < val_rand:
                    val_rand = new_val
                    improved = True
                else:
                    s_rand[i] = original
            
            if not improved:
                break
        
        if val_rand < best_val:
            best_val = val_rand
            best_signs = s_rand.copy()
    
    return best_signs
# EVOLVE-BLOCK-END

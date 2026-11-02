"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_score(x):
    """Compute the ratio: max|autoconv(x)| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    # Autoconvolution via FFT for speed
    n_fft = 1 << (2 * N - 1).bit_length()
    X = np.fft.rfft(x, n=n_fft)
    ac = np.fft.irfft(X * X, n=n_fft)[:2*N-1]
    # ac[k] = sum_i x[i]*x[i+k] for k=0..N-1 (and symmetric)
    # Actually irfft of |X|^2 gives circular conv, but with zero-padding it's linear
    # ac[0..N-1] are the relevant lags
    max_abs = np.max(np.abs(ac[:N]))
    return max_abs / (s * s)

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Best solution found
    best_x = np.ones(N)
    best_score = _compute_score(best_x)  # For all ones: max|ac| = N (at lag 0), sum=N, ratio = N/N^2 = 1/N
    
    # We want to minimize the ratio. All ones gives 1/N.
    # Let's try to find ±1 sequences that have lower peak autoconvolution.
    
    # Strategy: Use simulated annealing / local search on ±1 sequences
    # Objective: minimize max_k |sum_i x[i]x[i+k]| / (sum x)^2
    # For ±1 sequences, sum x can be small, making the denominator small,
    # but the numerator is bounded by N.
    # Actually, for balanced ±1 (sum=0), the denominator is 0, which is bad.
    # We need sum x to be non-zero.
    
    # Let's think about this differently.
    # If we use continuous values, we can scale. The ratio is scale-invariant
    # (numerator scales as t^2, denominator as t^2).
    # So we can fix the sum to be 1, and minimize max|autoconv|.
    
    # Alternative approach: Start with random ±1, then do local search
    # minimizing the peak absolute autoconvolution, while keeping sum non-zero.
    
    rng = np.random.default_rng(42)
    
    # Try multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initialize: random ±1 with slight bias to keep sum non-zero
        if restart == 0:
            x = np.ones(N)
        elif restart == 1:
            x = rng.choice([-1, 1], size=N)
            # Ensure sum is not too small
            if abs(np.sum(x)) < N * 0.05:
                x[0] = 1  # flip one to help
        else:
            x = rng.choice([-1, 1], size=N)
            if abs(np.sum(x)) < N * 0.05:
                # Flip a few to make sum larger
                idxs = rng.choice(N, size=min(10, N), replace=False)
                for i in idxs:
                    if np.sum(x) * x[i] < 0:
                        x[i] = -x[i]
        
        current_score = _compute_score(x)
        
        if current_score < best_score:
            best_score = current_score
            best_x = x.copy()
        
        # Local search: flip individual bits to reduce peak autoconv
        # Use a simple hill-climbing approach
        T = 0.1  # initial temperature
        T_min = 0.001
        cooling = 0.995
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            # Pick a random index to flip
            i = rng.integers(0, N)
            x[i] = -x[i]
            
            new_score = _compute_score(x)
            
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                # Accept with probability exp(-delta/T)
                delta = new_score - current_score
                if delta > 0 and rng.random() < np.exp(-delta / T):
                    current_score = new_score
                    no_improve += 1
                else:
                    x[i] = -x[i]  # revert
                    no_improve += 1
            
            T = max(T * cooling, T_min)
        
        if time.time() > deadline - 10:
            break
    
    # Final polish: try small perturbations around best solution
    if time.time() < deadline - 5:
        x = best_x.copy()
        current_score = best_score
        # Try flipping each bit and see if it helps
        for i in range(N):
            if time.time() > deadline - 5:
                break
            x[i] = -x[i]
            new_score = _compute_score(x)
            if new_score < current_score:
                current_score = new_score
                best_score = current_score
                best_x = x.copy()
            else:
                x[i] = -x[i]  # revert
    
    return best_x
# EVOLVE-BLOCK-END

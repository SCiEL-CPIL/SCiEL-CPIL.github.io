"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def _compute_score(f):
    """
    Computes the score for a given signed step function.
    Metric: Sum of cubes of absolute values of circular autocorrelation coefficients.
    This measures the 'flatness' of the autocorrelation. Lower is better.
    """
    fft_f = np.fft.fft(f)
    power_spectrum = fft_f * np.conj(fft_f)
    autocorr = np.fft.ifft(power_spectrum).real
    score = np.sum(np.abs(autocorr)**3)
    return score

def _local_search(f, max_steps=1000):
    """
    Perform local search (Hill Climbing) starting from f.
    Returns the best individual found and its score.
    """
    current = f.copy()
    current_score = _compute_score(current)
    
    for _ in range(max_steps):
        # Try flipping each bit? No, that's O(N^2 log N).
        # Try random flips until no improvement or max steps.
        # Actually, for N=400, checking all 400 neighbors is 400 FFTs.
        # 400 * (N log N) is fast enough for one step?
        # N=400, log N ~ 9. 400 * 400 * 9 ~ 1.4 million ops.
        # If we do this 1000 times, that's 1.4 billion ops. Might be too slow in Python loop.
        
        # Better: Random hill climbing.
        # Pick a random index, flip it. If better, keep. If not, try another.
        # Stop if no improvement in X attempts.
        
        improved = False
        attempts = 0
        max_attempts_no_improve = 50
        
        while attempts < max_attempts_no_improve:
            idx = np.random.randint(N)
            neighbor = current.copy()
            neighbor[idx] *= -1
            neighbor_score = _compute_score(neighbor)
            
            if neighbor_score < current_score:
                current = neighbor
                current_score = neighbor_score
                attempts = 0
                improved = True
            else:
                attempts += 1
        
        if not improved:
            break
            
    return current, current_score

def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Uses Iterated Local Search (ILS) to minimize the L3 norm of the circular autocorrelation.
    """
    # ILS Parameters
    max_iter = 50
    perturb_strength = 5 # Number of bits to flip for perturbation
    
    # Initialize with random signs
    current = np.random.choice([-1, 1], size=N)
    current, current_score = _local_search(current)
    
    best = current.copy()
    best_score = current_score
    
    for i in range(max_iter):
        # Perturb the best solution
        perturbed = best.copy()
        indices = np.random.choice(N, perturb_strength, replace=False)
        perturbed[indices] *= -1
        
        # Local search from perturbed state
        local_best, local_score = _local_search(perturbed)
        
        if local_score < best_score:
            best_score = local_score
            best = local_best.copy()
            
    return best
# EVOLVE-BLOCK-END

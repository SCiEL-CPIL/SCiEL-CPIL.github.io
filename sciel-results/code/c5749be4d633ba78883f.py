import time
import numpy as np

N = 600
DX = (0.5) / N

# Precompute FFT size
_M = 1
while _M < 2 * N - 1:
    _M *= 2

def _compute_score_fast(heights):
    """
    Fast computation of the normalized maximum of the autoconvolution.
    Assumes heights is already nonnegative and has length N.
    """
    n = len(heights)
    w = 0.5 / n
    
    f = np.zeros(_M, dtype=np.float64)
    f[:n] = heights
    
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=_M)
    
    max_conv = np.max(conv)
    total = np.sum(heights)
    
    if total <= 0:
        return np.inf
        
    score = max_conv * n / (total * total * w)
    return float(score)

def compute_score(heights):
    """
    Computes the normalized maximum of the autoconvolution.
    """
    heights = np.asarray(heights, dtype=np.float64)
    if np.any(heights < -1e-15):
        return np.inf
    heights = np.maximum(heights, 0.0)
    return _compute_score_fast(heights)

def _simulated_annealing(start_heights, start_temp, end_temp, n_steps, rng, time_budget):
    """
    Run simulated annealing starting from start_heights.
    Returns the best heights found and their score.
    """
    current = start_heights.copy()
    current_score = _compute_score_fast(current)
    
    best = current.copy()
    best_score = current_score
    
    # Precompute some constants
    n = len(current)
    
    for step in range(n_steps):
        if time.time() > time_budget:
            break
        
        # Propose a move: add Gaussian noise to a random subset of heights
        n_perturb = rng.integers(1, max(2, n // 10))
        indices = rng.integers(0, n, size=n_perturb)
        noise_scale = start_temp * (1.0 / n) * 0.1
        
        proposed = current.copy()
        proposed[indices] += noise_scale * rng.standard_normal(n_perturb)
        proposed = np.maximum(proposed, 0.0)
        
        proposed_score = _compute_score_fast(proposed)
        
        # Accept if better, or with probability exp(-delta/T)
        if proposed_score < current_score:
            current = proposed
            current_score = proposed_score
            if current_score < best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta = proposed_score - current_score
            if delta < 0:
                current = proposed
                current_score = proposed_score
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
            else:
                # Temperature decays geometrically
                temp = start_temp * (end_temp / start_temp) ** (step / max(1, n_steps))
                accept_prob = np.exp(-delta / (temp + 1e-12))
                if rng.random() < accept_prob:
                    current = proposed
                    current_score = proposed_score
    
    return best, best_score

def construct():
    # Deadline guard: 150s wall-clock limit, use 140s for safety
    start_time = time.time()
    deadline = start_time + 140
    
    # Initialize with uniform distribution
    uniform = np.ones(N, dtype=np.float64)
    uniform_score = _compute_score_fast(uniform)
    
    best_heights = uniform.copy()
    best_score = uniform_score
    
    # Multi-start simulated annealing
    rng = np.random.default_rng(42)
    
    n_starts = 20
    steps_per_start = 5000
    start_temp = 0.5
    end_temp = 0.001
    
    for i in range(n_starts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with some randomness
        if i == 0:
            init = uniform.copy()
        else:
            # Mix of uniform and random perturbations
            noise_level = 0.1 + 0.5 * (i / n_starts)
            init = uniform + noise_level * rng.standard_normal(N)
            init = np.maximum(init, 0.0)
        
        # Run simulated annealing
        local_best, local_score = _simulated_annealing(
            init, start_temp, end_temp, steps_per_start, rng, deadline - 5
        )
        
        if local_score < best_score:
            best_heights = local_best
            best_score = local_score
    
    # Final check: ensure we're within time budget
    if time.time() > deadline - 1:
        # Return best found so far
        pass
    
    return list(best_heights)

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def _overlap_max(h):
    """Compute max over all shifts of sum_i h_i * (1 - h_{i+shift}) for all shifts in 1..N-1."""
    h = np.asarray(h)
    n = len(h)
    max_overlap = 0.0
    for shift in range(1, n):
        # Compute sum_i h_i * (1 - h_{i+shift}) where indices wrap
        h_shifted = np.roll(h, shift)
        overlap = np.sum(h * (1.0 - h_shifted))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def construct():
    """
    Construct step heights h in [0, 1] with sum = N/2 to minimize max overlap of h with shifted 1-h.
    
    The overlap at shift k is sum_i h_i * (1 - h_{i+k mod N}).
    We want to minimize max_k overlap(k).
    
    Using Fourier analysis, the overlap at shift k can be expressed via the autocorrelation of h.
    For a stationary process, the maximum overlap is controlled by the power spectral density.
    
    Strategy: Use a smooth optimization approach. Start with a good initial guess and refine.
    """
    n = N
    
    # Initialize with alternating pattern (binary) as a starting point
    h = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
    
    # Try to improve using a simple local search / gradient-based approach
    # First, let's try a few structured patterns and pick the best
    
    candidates = []
    
    # Pattern 1: Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    
    # Pattern 2: Block pattern (e.g., 1,1,0,0,...)
    candidates.append(np.array([1.0 if (i // 2) % 2 == 0 else 0.0 for i in range(n)]))
    
    # Pattern 3: More balanced - use a sinusoidal pattern
    for freq in [1, 2, 3, 5, 7]:
        t = np.linspace(0, 2 * np.pi * freq, n, endpoint=False)
        h_sin = 0.5 + 0.5 * np.cos(t)
        # Normalize to sum = n/2
        h_sin = h_sin * (n / 2) / np.sum(h_sin)
        h_sin = np.clip(h_sin, 0, 1)
        # Re-normalize after clipping
        h_sin = h_sin * (n / 2) / np.sum(h_sin)
        candidates.append(h_sin)
    
    # Pattern 4: Random with constraint
    np.random.seed(42)
    for _ in range(50):
        h_rand = np.random.uniform(0, 1, n)
        h_rand = h_rand * (n / 2) / np.sum(h_rand)
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = h_rand * (n / 2) / np.sum(h_rand)
        candidates.append(h_rand)
    
    # Evaluate all candidates and pick the best
    best_h = None
    best_score = np.inf
    
    for cand in candidates:
        # Ensure sum is exactly n/2 (within floating point tolerance)
        cand = cand.copy()
        diff = n / 2 - np.sum(cand)
        if abs(diff) > 1e-10:
            # Distribute the difference
            cand += diff / n
            cand = np.clip(cand, 0, 1)
            # Final adjustment
            diff = n / 2 - np.sum(cand)
            cand += diff / n
        
        score = _overlap_max(cand)
        if score < best_score:
            best_score = score
            best_h = cand.copy()
    
    if best_h is None:
        # Fallback
        best_h = np.full(n, 0.5)
    
    # Try to refine the best candidate using a simple hill-climbing approach
    # with small perturbations
    current = best_h.copy()
    current_score = best_score
    best_overall = current.copy()
    best_overall_score = current_score
    
    # Simple local optimization
    step_size = 0.1
    for it in range(200):
        # Try perturbing a random element
        idx = np.random.randint(n)
        old_val = current[idx]
        new_val = old_val + np.random.uniform(-step_size, step_size)
        new_val = np.clip(new_val, 0, 1)
        
        if new_val == old_val:
            continue
        
        # Adjust other elements to maintain sum
        diff = old_val - new_val
        if abs(diff) < 1e-12:
            continue
        
        # Distribute diff to other random elements
        other_idx = np.random.choice([i for i in range(n) if i != idx], size=5, replace=False)
        current[idx] = new_val
        for j, oidx in enumerate(other_idx):
            adj = diff / (len(other_idx) - j)
            current[oidx] = np.clip(current[oidx] + adj, 0, 1)
        
        # Final sum correction
        sum_err = n / 2 - np.sum(current)
        if abs(sum_err) > 1e-10:
            current += sum_err / n
            current = np.clip(current, 0, 1)
            sum_err = n / 2 - np.sum(current)
            current += sum_err / n
        
        new_score = _overlap_max(current)
        
        if new_score < current_score - 1e-10:
            current_score = new_score
            if new_score < best_overall_score:
                best_overall_score = new_score
                best_overall = current.copy()
            step_size = max(0.01, step_size * 0.95)
        else:
            # Revert
            current[idx] = old_val
            current[other_idx] = current[other_idx] - np.array([diff / (len(other_idx) - j) for j in range(len(other_idx))])
            # Simpler: just recompute from best
            current = best_overall.copy()
            current_score = best_overall_score
            step_size = max(0.01, step_size * 0.9)
    
    return best_overall
# EVOLVE-BLOCK-END

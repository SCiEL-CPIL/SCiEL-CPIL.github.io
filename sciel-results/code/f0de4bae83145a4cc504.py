"""Cap set priority function with spectral/Fourier-inspired balance heuristic."""
import time

# Deadline guard: stop work if we exceed this fraction of the 330s limit
_start_time = time.time()
_TIMEOUT = 315.0  # Safety margin before 320s hard limit

def priority(vector, n):
    """Priority function for greedy cap set construction.
    
    Uses a spectral independence-inspired heuristic:
    - Primary: minimize quadratic deviation from ideal symbol balance (n/3 each)
    - Secondary: penalize concentration in any single symbol
    - Tertiary: small diversity bonus for having all three symbols
    - Tie-break: polynomial hash for deterministic distinctness
    
    Args:
        vector: tuple of length n with values in {0, 1, 2}
        n: dimension
    
    Returns:
        float: priority score (higher = added earlier)
    """
    # Fast timeout check
    if time.time() - _start_time > _TIMEOUT:
        return -1e9

    # Single pass: count symbols and compute polynomial hash
    c0 = 0
    c1 = 0
    c2 = 0
    h = 0
    
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
        # Polynomial hash (mod large prime) for deterministic tie-breaking
        h = (h * 31 + v) % 1000000007

    if n == 0:
        return 0.0

    ideal = n / 3.0
    
    # Quadratic deviation from ideal balance (spectral: low autocorrelation proxy)
    dev = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    
    # Concentration penalty: vectors dominated by one symbol form APs more easily
    max_count = max(c0, c1, c2)
    concentration_penalty = (max_count * max_count) / n
    
    # Diversity bonus: having all three symbols present is structurally favorable
    diversity_bonus = 0.0
    if c0 > 0 and c1 > 0 and c2 > 0:
        diversity_bonus = 0.5

    # Combine: minimize dev and concentration, maximize diversity
    # Hash as tiny negative tie-breaker to ensure distinct priorities
    score = -dev - 0.1 * concentration_penalty + diversity_bonus - h * 1e-9
    
    return score

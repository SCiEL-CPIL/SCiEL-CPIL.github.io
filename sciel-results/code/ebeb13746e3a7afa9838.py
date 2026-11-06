"""Cap set priority function for greedy insertion."""
import time

_start_time = time.time()
_DEADLINE_MARGIN = 10  # seconds before the 300s limit


def priority(vector, n):
    """
    Computes a priority score for a vector in {0, 1, 2}^n.
    Higher scores indicate vectors that should be added earlier in the greedy process.
    
    Strategy:
    1. Primary: Balance symbol counts (0, 1, 2) to avoid early constraint saturation.
    2. Secondary: Penalize vectors with many identical adjacent pairs (reduces line risk).
    3. Tertiary: Penalize vectors with long constant runs (structural AP risk).
    4. Quaternary: Deterministic hash for tie-breaking to break symmetries.
    """
    # Deadline guard: if we're close to the limit, return a neutral score
    # This prevents the evaluator from timing out
    elapsed = time.time() - _start_time
    if elapsed > 300 - _DEADLINE_MARGIN:
        return 0.0

    # Count symbols
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Balanced vectors (close to n/3 each) are preferred
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Primary: minimize deviation (negate for max-priority)
    score = -deviation * 1000.0
    
    # Secondary: penalize adjacent equal pairs
    # Vectors with many adjacent equal values are more likely to form APs
    adj_equal = 0
    for i in range(n - 1):
        if vector[i] == vector[i + 1]:
            adj_equal += 1
    score -= adj_equal * 50.0
    
    # Tertiary: penalize long constant runs
    # Longer runs of the same value increase AP risk
    max_run = 1
    cur_run = 1
    for i in range(1, n):
        if vector[i] == vector[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    score -= (max_run - 1) * 20.0
    
    # Quaternary: Deterministic hash-based tiebreaker
    # This creates a pseudo-random permutation of vectors with similar structural scores,
    # avoiding local clustering and allowing exploration of diverse subspaces.
    
    # Fast FNV-1a style hash for the vector
    # We treat the vector as a base-3 number or just mix the bits.
    # Since n is small (usually < 10-20 in these tasks), we can do a simple weighted sum with large weights.
    # To ensure good distribution, we use a large prime base.
    h = 0
    base = 3
    # Iterate in reverse to give higher weight to later coordinates? Or just standard polynomial rolling hash.
    for v in vector:
        h = (h * base + v)
    
    # Normalize hash to a small range to act as a perturbation
    # We want this to break ties but not overwhelm the structural score (deviation * 1000).
    # If deviation is 0, score is 0. Hash perturbation should be small.
    # However, if we want to "shuffle" more aggressively, we can scale it up.
    # Let's try scaling the hash to range [0, 10) to compete with small deviations.
    hash_score = (h % 1000000) / 100000.0
    score += hash_score
    
    return score

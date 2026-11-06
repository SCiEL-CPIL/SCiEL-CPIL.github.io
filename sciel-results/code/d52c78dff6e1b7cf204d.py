"""Find a dissociated reciprocal set from [1,22] maximizing cardinality."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    # We want the largest k such that some k-subset of [1,22] has all distinct subset sums.
    # Search k from 22 down to 1.
    
    n = 22
    best_result = [1]  # fallback: single element is always valid
    
    for k in range(n, 0, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try all k-combinations, but with early termination
        for combo in combinations(range(1, n + 1), k):
            if time.time() > deadline:
                break
            if _is_dissociated(combo):
                best_result = list(combo)
                found = True
                break
        
        if found:
            break
    
    return best_result


def _is_dissociated(elements):
    """Check if all subset sums of reciprocals are distinct."""
    k = len(elements)
    reciprocals = [1.0 / x for x in elements]
    
    # Generate all 2^k subset sums
    # Use a set to detect collisions
    seen = set()
    # Start with empty sum = 0
    seen.add(0.0)
    
    # Build subset sums iteratively
    # sums[i] = set of all subset sums using first i elements
    # But that's 2^k * k which is fine for k <= 12 or so
    # For larger k, we might need a different approach
    
    if k > 20:
        # Too many subset sums to enumerate naively; use a different check
        # Actually 2^20 = 1M, which is feasible but slow in Python
        # Let's just do it with numpy for speed
        import numpy as np
        # Generate all 2^k bitmasks
        # For k up to 22, 2^22 = 4M, too slow in pure Python
        # Use the fact that if k is large, collisions are likely
        # Use a heuristic: check random subset of pairs? No, must be exact.
        # For now, limit to k where 2^k is manageable
        # 2^20 = 1,048,576 - feasible with numpy
        # 2^22 = 4,194,304 - might be too slow
        
        # Use incremental approach with numpy
        sums = np.array([0.0])
        for r in reciprocals:
            sums = np.concatenate([sums, sums + r])
        # Check for duplicates
        # Sort and check for adjacent equal
        sorted_sums = np.sort(sums)
        diffs = np.diff(sorted_sums)
        return np.all(diffs > 1e-12)
    
    # For k <= 20, use incremental set approach
    sums = [0.0]
    for r in reciprocals:
        new_sums = [s + r for s in sums]
        sums = sums + new_sums
        # Early check: if len(set(sums)) < len(sums), collision
        if len(set(sums)) < len(sums):
            return False
    
    return len(set(sums)) == len(sums)
# EVOLVE-BLOCK-END

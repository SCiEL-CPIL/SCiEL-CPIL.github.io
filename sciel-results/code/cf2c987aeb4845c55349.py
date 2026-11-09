"""Find maximum cardinality dissociated reciprocal subset of [1,22]."""
import time
import itertools

def _is_dissociated(subset):
    """Check if all reciprocal subset sums are distinct."""
    n = len(subset)
    if n == 0:
        return True
    # Compute all 2^n subset sums of 1/x for x in subset
    # Use iterative doubling
    sums = [0.0]
    for x in subset:
        inv = 1.0 / x
        sums += [s + inv for s in sums]
    # Check all distinct
    if len(set(sums)) != len(sums):
        return False
    return True

def _is_dissociated_fast(subset):
    """Faster check using sorted comparison."""
    n = len(subset)
    sums = [0.0]
    for x in subset:
        inv = 1.0 / x
        sums += [s + inv for s in sums]
    sums.sort()
    for i in range(1, len(sums)):
        if sums[i] == sums[i-1]:
            return False
    return True

def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Start with the known good set of 9
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best)
    
    # Try to extend to size 10, 11, ...
    # Strategy: greedy addition with backtracking
    # For each candidate size, try all combinations
    
    # First, verify the seed
    if not _is_dissociated_fast(best):
        # Fallback: find a valid set
        for size in range(9, 0, -1):
            found = False
            for combo in itertools.combinations(range(1, 23), size):
                if _is_dissociated_fast(list(combo)):
                    best = list(combo)
                    best_size = size
                    found = True
                    break
            if found:
                break
    
    # Now try to find larger sets
    # For size 10: try adding one element to the seed
    candidates = list(range(1, 23))
    
    # Try sizes from 10 upward
    for target_size in range(best_size + 1, 14):
        if time.time() > deadline:
            break
        
        # Use greedy: try to build a set of target_size
        # Start from best and try to swap/add
        if target_size == best_size + 1:
            # Try adding one element
            for elem in candidates:
                if elem in best:
                    continue
                new_set = best + [elem]
                if _is_dissociated_fast(new_set):
                    best = new_set
                    best_size = target_size
                    break
        else:
            # For larger jumps, use a more systematic approach
            # Try random restarts with local search
            import random
            random.seed(42)
            found = False
            for attempt in range(5000):
                if time.time() > deadline:
                    break
                # Random subset of target_size
                sub = random.sample(candidates, target_size)
                if _is_dissociated_fast(sub):
                    best = sub
                    best_size = target_size
                    found = True
                    break
            
            if found:
                # Try to extend further
                continue
            else:
                # Can't find this size, try smaller
                break
    
    return best

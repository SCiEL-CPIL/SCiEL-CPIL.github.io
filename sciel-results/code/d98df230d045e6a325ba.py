"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_validity(s):
    """Check if set s has cardinality-separated subset sums."""
    n = len(s)
    if n == 0:
        return True
    # sums[k] = set of all sums using exactly k elements from s
    sums = [set() for _ in range(n + 1)]
    sums[0].add(0)
    
    for x in s:
        # Update from high to low to avoid using x multiple times
        for k in range(n, 0, -1):
            if sums[k-1]:
                for prev_sum in sums[k-1]:
                    sums[k].add(prev_sum + x)
    
    # Check that all sums[k] are pairwise disjoint
    seen = set()
    for k in range(n + 1):
        for val in sums[k]:
            if val in seen:
                return False
            seen.add(val)
    return True

def _greedy_construct(max_val=256, max_size=60, time_limit=55):
    """Greedy construction: try to build a valid set by adding elements."""
    start_time = time.time()
    
    # Try different strategies
    best_set = []
    
    # Strategy 1: Greedy from large to small
    for strategy in ['desc', 'asc', 'random']:
        if time.time() - start_time > time_limit:
            break
        
        candidates = list(range(1, max_val + 1))
        if strategy == 'desc':
            candidates.sort(reverse=True)
        elif strategy == 'asc':
            candidates.sort()
        else:
            random.shuffle(candidates)
        
        current_set = []
        # sums[k] for current_set
        n_curr = 0
        sums = [set() for _ in range(max_size + 1)]
        sums[0].add(0)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            if len(current_set) >= max_size:
                break
            
            # Try adding x
            # New sums would be: for each k, new_sums[k] = sums[k] ∪ {s + x for s in sums[k-1]}
            # Check if this creates any collision across cardinalities
            
            # Compute what the new sums would look like
            # We need to check: for each k, the new elements added to sums[k] 
            # (which are {s+x for s in sums[k-1]}) don't collide with any existing sums[j] for j≠k
            # AND don't collide with each other within the same k (but that's fine, same k collisions are allowed)
            
            # Actually, the constraint is: sums[k] and sums[l] must be disjoint for k≠l
            # When we add x, for each k, we add {s+x : s in sums[k-1]} to sums[k]
            # These new values must not be in any sums[j] for j≠k (in the final state)
            
            # Efficient check: for each k, compute new_vals = {s+x for s in sums[k-1]}
            # Check new_vals ∩ (union of sums[j] for j≠k, in the updated state) = empty
            # But the updated state also has new values in other cardinalities...
            
            # Simpler: just do the full check on current_set + [x]
            test_set = current_set + [x]
            if _check_validity(test_set):
                # Update sums incrementally
                for k in range(len(current_set), 0, -1):
                    if sums[k-1]:
                        for prev_sum in sums[k-1]:
                            sums[k].add(prev_sum + x)
                current_set.append(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
    
    # Strategy 2: Random restarts with local search
    if time.time() - start_time < time_limit - 5:
        for _ in range(10):
            if time.time() - start_time > time_limit - 2:
                break
            
            random.shuffle(candidates)
            current_set = []
            for x in candidates:
                if time.time() - start_time > time_limit - 2:
                    break
                if len(current_set) >= max_size:
                    break
                test_set = current_set + [x]
                if _check_validity(test_set):
                    current_set.append(x)
            
            if len(current_set) > len(best_set):
                best_set = current_set[:]
    
    return best_set

def construct():
    """Construct a valid set with maximum size."""
    start_time = time.time()
    
    # Quick fallback
    result = _greedy_construct(max_val=256, max_size=60, time_limit=55)
    
    # Ensure we return a valid list of ints
    if not result:
        # Fallback: use powers of 2 which always work
        result = [2**i for i in range(9) if 2**i <= 256]
    
    # Verify validity
    if not _check_validity(result):
        # If somehow invalid, use the safe fallback
        result = [2**i for i in range(9) if 2**i <= 256]
    
    # Convert to ensure Python int types
    return [int(x) for x in result]
# EVOLVE-BLOCK-END

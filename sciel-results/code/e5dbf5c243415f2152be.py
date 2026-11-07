"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _is_valid_fast(els):
    """Check if all non-empty subset sums are distinct and pairwise incomparable under divisibility."""
    n = len(els)
    if n == 0:
        return True
    
    # Generate all subset sums
    sums = set()
    # Use bitmasks or iterative expansion
    current_sums = {0}
    for e in els:
        new_sums = set()
        for s in current_sums:
            ns = s + e
            if ns in sums or ns in new_sums:
                # If we find a duplicate sum, it's invalid
                # Note: if s=0, ns=e. If e is already in sums, invalid.
                # If ns is in current_sums (from previous elements) or new_sums (from current element), it's a collision.
                # Actually, if we maintain a global set of all generated sums so far:
                pass
        # Let's stick to a simple robust method for n<=15
        pass

    # Efficient generation
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        # Unroll or use bit ops?
        # For small n, simple loop is fine.
        for i in range(n):
            if mask & (1 << i):
                s += els[i]
        if s in sums:
            return False
        sums.add(s)
        
    sum_list = sorted(sums)
    m = len(sum_list)
    # Check divisibility
    # For each sum s, check if any other sum is a multiple of s
    # Optimization: if s > max_sum / 2, no multiple exists in the set (except 1*s which is itself, but we check distinct values)
    max_sum = sum_list[-1]
    
    # Create a set for O(1) lookup
    sum_set = set(sum_list)
    
    for s in sum_list:
        if s == 0: continue
        # Check multiples 2s, 3s, ... up to max_sum
        # If 2*s > max_sum, break
        mult = 2 * s
        while mult <= max_sum:
            if mult in sum_set:
                return False
            mult += s
            
    return True

def _greedy_construct(max_val=256, target_size=12, timeout=55):
    start = time.time()
    best_size = 0
    best_set = []
    
    # We want to find a large set.
    # Strategy: Try to build a set by adding elements one by one from a pool of candidates.
    # To increase chance of success, pick candidates from a range that minimizes divisibility risks.
    # High numbers are better?
    
    for attempt in range(100):
        if time.time() - start > timeout - 2:
            break
        
        # Generate a pool of candidates
        # Use a random subset of [1, 256] but biased towards higher values?
        # Or just random sample.
        pool = random.sample(range(1, max_val + 1), 30)
        
        current_set = []
        # Shuffle pool to try different orders
        random.shuffle(pool)
        
        for cand in pool:
            if time.time() - start > timeout - 2:
                break
            new_set = current_set + [cand]
            if _is_valid_fast(new_set):
                current_set = new_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    if best_size >= target_size:
                        return best_set
        
    return best_set

def construct():
    result = _greedy_construct()
    # Ensure we return at least a valid small set if greedy fails completely (unlikely)
    if not result:
        result = [31, 32, 34]
    return [int(x) for x in result]
# EVOLVE-BLOCK-END

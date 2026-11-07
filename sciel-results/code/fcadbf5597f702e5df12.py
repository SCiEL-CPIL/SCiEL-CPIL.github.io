import time
import random
import sys

# EVOLVE-BLOCK-START

def _check_antichain(sums):
    """
    Check if the set of sums is an antichain under divisibility.
    i.e., no two distinct elements divide each other.
    """
    if not sums:
        return True
    sorted_sums = sorted(sums)
    max_sum = sorted_sums[-1]
    # Use a boolean list for fast lookup
    # Max possible sum for 10 elements from 1..256 is 10*256 = 2560
    # But we can just size it to max_sum + 1
    is_sum = [False] * (max_sum + 1)
    for s in sorted_sums:
        is_sum[s] = True
        
    for s in sorted_sums:
        multiple = s * 2
        while multiple <= max_sum:
            if is_sum[multiple]:
                return False
            multiple += s
    return True

def _can_add(current_sums, candidate, max_val=256):
    """
    Check if adding 'candidate' to the set corresponding to 'current_sums'
    results in a valid antichain of sums.
    Returns the new set of sums if valid, else None.
    """
    # New sums are current_sums U {s + candidate for s in current_sums} U {candidate}
    # Note: 0 is not in current_sums, but we treat it as a base sum for the new element
    # Actually, subset sums of A U {c} are S_A U {s+c | s in S_A} U {c}
    # Wait, S_A contains sums of non-empty subsets of A.
    # The new sums are:
    # 1. Old sums (subsets not containing c)
    # 2. c (subset {c})
    # 3. s + c (subsets containing c, where s is sum of subset of A, possibly empty)
    # So we need to consider s=0 in the "s+c" part to get c.
    
    new_sums = set(current_sums)
    new_sums.add(candidate)
    
    # Generate sums with candidate
    # We iterate over existing sums and add candidate
    # Also add candidate + 0 = candidate (already added)
    for s in current_sums:
        new_sums.add(s + candidate)
    
    # Check antichain property
    if not _check_antichain(new_sums):
        return None
    
    return new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # We want to maximize size. Theoretical max is 10.
    # Let's try to find a set of size 10.
    # If we can't, we try 9, etc.
    
    # Heuristic: Larger numbers are less likely to divide smaller numbers?
    # Actually, if we pick large numbers, their sums are large.
    # Smaller numbers are "dangerous" because they can divide many things.
    # Let's try to construct from larger numbers downwards?
    # Or just random restarts.
    
    best_set = []
    best_size = 0
    
    # Try to find size 10
    target_size = 10
    
    # Random restarts
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        # Removed invalid shuffle of range object
        
        current_set = []
        current_sums = set()
        
        # Candidates to try
        # Use descending order as a base heuristic, then shuffle
        candidates = list(range(256, 0, -1))
        random.shuffle(candidates)
        
        success = True
        for c in candidates:
            if len(current_set) >= target_size:
                break
                
            # Check if we can add c
            new_sums = _can_add(current_sums, c)
            if new_sums is not None:
                current_set.append(c)
                current_sums = new_sums
                
        if len(current_set) == target_size:
            # Verify one last time
            if _check_antichain(current_sums):
                return current_set
        
        # If we didn't reach target size, update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # If best_size is 10, we would have returned.
        # If we are lucky, we might find it quickly.
        
    # If we didn't find size 10, return best found
    # But we should ensure best_set is valid.
    if best_set:
        # Verify best_set one last time
        # Reconstruct sums
        sums = set()
        n = len(best_set)
        for i in range(1, 1 << n):
            s = 0
            for j in range(n):
                if i & (1 << j):
                    s += best_set[j]
            sums.add(s)
        if _check_antichain(sums):
            return best_set
        
    # Fallback: return a single element which is always valid
    return [1]

# EVOLVE-BLOCK-END

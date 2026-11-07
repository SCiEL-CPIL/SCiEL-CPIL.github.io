import time
import random

DEADLINE_SECONDS = 85.0

def _check_divisibility_fast(sums):
    """
    Check if no two distinct subset sums divide one another.
    sums: a set of integers.
    """
    if not sums:
        return True
    
    max_sum = max(sums)
    sum_set = sums # Already a set
    
    for a in sums:
        if a == 0:
            continue
        # Check multiples of a
        multiple = a * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += a
            
    return True

def _all_subset_sums(elements):
    """Generate all nonempty subset sums."""
    sums = set()
    for e in elements:
        new_sums = set()
        for s in sums:
            ns = s + e
            if ns not in sums:
                new_sums.add(ns)
        new_sums.add(e)
        sums.update(new_sums)
    return sums

def _is_valid(elements):
    """Check if the set of elements forms a valid configuration."""
    if not elements:
        return True
    sums = _all_subset_sums(elements)
    return _check_divisibility_fast(sums)

def _try_add(elements, candidate):
    """
    Try to add candidate to elements. Returns True if valid, False otherwise.
    This is a fast check that only considers the new sums created by 'candidate'.
    """
    # Get current sums
    sums = _all_subset_sums(elements)
    
    # New sums are: candidate, and candidate + s for all s in sums
    # We need to check if any of these new sums divide or are divisible by existing sums.
    # Actually, we just need to check if the NEW set of sums is valid.
    # But we can be smarter:
    # 1. Check if candidate divides any existing sum or is divisible by any existing sum.
    # 2. Check if (candidate + s1) divides (candidate + s2) or existing sums, etc.
    
    # Given the complexity, it's often faster to just compute the new full set and check.
    # But for n < 15, computing the new set is fast.
    # Let's just compute the new set and use the fast checker.
    new_elements = elements + [candidate]
    new_sums = _all_subset_sums(new_elements)
    return _check_divisibility_fast(new_sums)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    all_candidates = list(range(1, 257))
    random.seed(42)
    
    best_set = []
    best_size = 0
    
    # Initial Greedy Construction
    # Sort candidates in descending order to pack large numbers first?
    # Or random order?
    # Large numbers have fewer multiples, so they are "safer" to add.
    candidates = all_candidates[:]
    random.shuffle(candidates)
    
    current_set = []
    
    # Try to add as many as possible
    for c in candidates:
        if time.time() > deadline - 10:
            break
        if _try_add(current_set, c):
            current_set.append(c)
    
    if len(current_set) > best_size:
        best_set = list(current_set)
        best_size = len(current_set)
        
    # Local Search / Simulated Annealing
    # We will try to improve the size.
    # Move: Add a random candidate.
    # Move: Swap a current element with a random candidate.
    
    current = list(best_set)
    current_set_obj = set(current)
    
    # Pre-compute sums for current set to speed up checks?
    # The _try_add function recomputes sums. This is expensive.
    # Let's optimize by maintaining the sum set.
    
    def get_sums(elements):
        return _all_subset_sums(elements)
    
    current_sums = get_sums(current)
    
    for _ in range(10000): # Max iterations
        if time.time() > deadline - 5:
            break
            
        # Try to add a random element
        added = False
        if len(current) < 30:
            c = random.choice(all_candidates)
            if c not in current_set_obj:
                # Fast check: does c conflict with current_sums?
                # We need to check if adding c creates any divisibility.
                # New sums: c, c+s for s in current_sums.
                # Let's just do the full check for simplicity first, 
                # but we can optimize _try_add to use current_sums.
                
                # Optimization:
                # Check if c divides any s in current_sums or vice versa.
                # Check if (c+s1) divides (c+s2) or any s in current_sums.
                # This is still complex. Let's stick to full check but with fast validator.
                
                if _try_add(current, c):
                    current.append(c)
                    current_set_obj.add(c)
                    current_sums = get_sums(current)
                    if len(current) > best_size:
                        best_set = list(current)
                        best_size = len(current)
                    added = True
        
        if not added:
            # Try to swap
            if current:
                remove_idx = random.randint(0, len(current) - 1)
                removed = current.pop(remove_idx)
                current_set_obj.discard(removed)
                
                # Try to add a new element
                success = False
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    c = random.choice(all_candidates)
                    if c not in current_set_obj and c != removed:
                        if _try_add(current, c):
                            current.append(c)
                            current_set_obj.add(c)
                            current_sums = get_sums(current)
                            if len(current) > best_size:
                                best_set = list(current)
                                best_size = len(current)
                            success = True
                            break
                
                if not success:
                    # Revert
                    current.append(removed)
                    current_set_obj.add(removed)
                    current_sums = get_sums(current)

    # Final verification
    if best_set and not _is_valid(best_set):
        # Fallback
        best_set = [31, 32, 34]
        
    return [int(x) for x in best_set]

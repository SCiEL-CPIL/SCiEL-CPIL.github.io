from fractions import Fraction
import time
import random

def construct():
    start_time = time.time()
    timeout = 55  # seconds, leave margin
    
    # Precompute reciprocals
    recips = {i: Fraction(1, i) for i in range(1, 25)}
    
    # Greedy construction: add elements from largest to smallest
    # This tends to be safer as reciprocals are smaller and less likely to sum to others
    S = set()
    # We will use a faster validity check for the greedy phase
    for a in range(24, 0, -1):
        if time.time() - start_time > timeout:
            break
        if _is_valid_fast(S, a, recips, timeout - (time.time() - start_time)):
            S.add(a)
            
    best_S = set(S)
    best_size = len(S)
    
    # Local Search: try to add elements we skipped, or swap
    # We use a randomized local search to escape local optima
    random.seed(42)
    
    while time.time() - start_time < timeout - 5:
        improved = False
        
        # Try to add any missing element
        candidates = [x for x in range(1, 25) if x not in S]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() - start_time > timeout - 5:
                break
            if _is_valid_fast(S, x, recips, timeout - (time.time() - start_time)):
                S.add(x)
                improved = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                break  # Restart the loop with the new S
        
        if not improved:
            # Try swaps: remove one, add another
            # Only if we have time
            if time.time() - start_time > timeout - 10:
                break
            
            s_elements = list(S)
            random.shuffle(s_elements)
            c_elements = [x for x in range(1, 25) if x not in S]
            random.shuffle(c_elements)
            
            swap_found = False
            for y in s_elements:
                if time.time() - start_time > timeout - 5:
                    break
                for z in c_elements:
                    if time.time() - start_time > timeout - 5:
                        break
                    new_S = set(S)
                    new_S.remove(y)
                    new_S.add(z)
                    # Check validity of new_S
                    # This is expensive, so we limit attempts
                    if _is_valid_full(new_S, recips, timeout - (time.time() - start_time)):
                        S = new_S
                        swap_found = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        break
                if swap_found:
                    break
            
            # If no swap improved size, we might be stuck. 
            # We can try to just keep the best_S so far.
            # Since swaps keep size constant, they don't increase cardinality directly,
            # but they might enable future additions.
            # To keep it simple and fast, we'll just rely on additions.
            # If no additions possible, we break.
            if not improved and not swap_found:
                # Try a few random restarts? No, too slow.
                # Just break if we can't add anything.
                pass

    return sorted(best_S)

def _is_valid_fast(S, x, recips, remaining_time=55):
    """
    Checks if adding x to S (which is assumed valid) results in a valid set.
    1. 1/x is not a subset sum of S.
    2. For all a in S, 1/a - 1/x is not a subset sum of S \ {a}.
    """
    start = time.time()
    S = set(S)
    if not S:
        return True
        
    # 1. Check if 1/x is in subset sums of S
    target_x = recips[x]
    sums_S = _get_subset_sums(S, recips, remaining_time)
    if sums_S is None: # Timeout
        return True
    if target_x in sums_S:
        return False
        
    # 2. Check existing elements
    for a in S:
        if time.time() - start > remaining_time:
            return True
        target_a = recips[a]
        diff = target_a - target_x
        if diff <= 0:
            continue # Sum of positive reciprocals cannot be <= 0
        # We need to check if diff is in subset sums of S \ {a}
        # To do this efficiently, we can generate subset sums of S \ {a}
        # But doing this for every a is O(N * 2^N)
        # Optimization: If N is small, it's fine.
        S_minus_a = S - {a}
        sums_minus_a = _get_subset_sums(S_minus_a, recips, remaining_time - (time.time() - start))
        if sums_minus_a is None:
            return True
        if diff in sums_minus_a:
            return False
            
    return True

def _get_subset_sums(S, recips, remaining_time=55):
    """
    Returns a set of all subset sums of reciprocals of elements in S.
    """
    start = time.time()
    S_list = list(S)
    sums = {Fraction(0)}
    for b in S_list:
        if time.time() - start > remaining_time:
            return None
        new_sums = set()
        rb = recips[b]
        for s in sums:
            new_sums.add(s + rb)
        sums.update(new_sums)
    return sums

def _is_valid_full(S, recips, remaining_time=55):
    """
    Checks if S is a valid set.
    """
    start = time.time()
    S = set(S)
    if not S:
        return True
    for a in S:
        if time.time() - start > remaining_time:
            return True
        others = S - {a}
        target = recips[a]
        sums = _get_subset_sums(others, recips, remaining_time - (time.time() - start))
        if sums is None:
            return True
        if target in sums:
            return False
    return True

import time
import random

def _get_sums(elements):
    """Get all non-empty subset sums as a set."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_valid_fast(elements):
    """Check if the set satisfies all constraints using fast multiple check."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not (1 <= x <= 256):
            return False
    
    sums = _get_sums(elements)
    sums_set = set(sums)
    max_sum = max(sums_set)
    
    for s in sums_set:
        # Check if any multiple of s is in the set
        # k*s is in sums_set for k >= 2
        max_mult = max_sum // s
        for k in range(2, max_mult + 1):
            if s * k in sums_set:
                return False
    return True

def _check_addition_fast(current_elements, candidate, current_sums_set, current_max_sum):
    """Check if adding candidate maintains validity. Returns (new_sums_set, new_max_sum) if valid, None otherwise."""
    new_sums_set = set(current_sums_set)
    new_max_sum = current_max_sum
    
    for s in current_sums_set:
        ns = s + candidate
        new_sums_set.add(ns)
        if ns > new_max_sum:
            new_max_sum = ns
            
    new_sums_set.add(candidate)
    if candidate > new_max_sum:
        new_max_sum = candidate
        
    # Fast validity check on the new set
    for s in new_sums_set:
        max_mult = new_max_sum // s
        for k in range(2, max_mult + 1):
            if s * k in new_sums_set:
                return None
                
    return new_sums_set, new_max_sum

def _try_strategy_fast(start_elements, candidate_order, deadline):
    """Try a construction strategy using fast checks."""
    if time.time() > deadline:
        return list(start_elements)
    
    elements = list(start_elements)
    sums_set = _get_sums(elements)
    max_sum = max(sums_set) if sums_set else 0
    
    for candidate in candidate_order:
        if time.time() > deadline:
            break
        if candidate in elements:
            continue
        res = _check_addition_fast(elements, candidate, sums_set, max_sum)
        if res is not None:
            sums_set, max_sum = res
            elements.append(candidate)
    
    return elements

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_result = [31, 32, 34]
    
    all_candidates = list(range(1, 257))
    
    random.seed(42)
    
    # Strategy 1: Greedy from various starts
    # Focus on larger numbers which have fewer multiples
    for start_size in range(1, 6):
        if time.time() > deadline:
            break
            
        for _ in range(50): # 50 trials per start size
            if time.time() > deadline:
                break
                
            start_elements = random.sample(all_candidates, start_size)
            start_elements.sort()
            
            # Check if start is valid
            if not _is_valid_fast(start_elements):
                continue
                
            # Try to grow
            remaining = [x for x in all_candidates if x not in start_elements]
            # Heuristic: try larger numbers first
            random.shuffle(remaining)
            
            result = _try_strategy_fast(start_elements, remaining, deadline)
            if len(result) > len(best_result):
                best_result = result
                
    # Strategy 2: Local Search / Hill Climbing
    # Try to improve a random valid set by swapping elements
    for _ in range(100): # 100 iterations
        if time.time() > deadline:
            break
            
        # Start with a random set of size close to best
        target_size = min(len(best_result) + 1, 12)
        current_set = random.sample(all_candidates, target_size)
        current_set.sort()
        
        if not _is_valid_fast(current_set):
            # Try to fix it by removing elements? Or just skip.
            # Let's try a smaller random set and grow
            current_set = random.sample(all_candidates, 5)
            current_set.sort()
            if not _is_valid_fast(current_set):
                continue
                
            sums_set = _get_sums(current_set)
            max_sum = max(sums_set)
            
            # Grow greedily
            remaining = [x for x in all_candidates if x not in current_set]
            random.shuffle(remaining)
            
            for candidate in remaining:
                if time.time() > deadline:
                    break
                res = _check_addition_fast(current_set, candidate, sums_set, max_sum)
                if res is not None:
                    sums_set, max_sum = res
                    current_set.append(candidate)
            
            if len(current_set) > len(best_result):
                best_result = current_set
                continue
            
            # If we didn't beat best, try local search on this valid set
            # Try swapping one element for another to see if we can add more later?
            # This is complex. Let's just do more random restarts.
            
    # Final validation
    if _is_valid_fast(best_result):
        return [int(x) for x in best_result]
    
    # Fallback
    return [31, 32, 34]

import time
import random

def _subset_sums_set(elements):
    """Get all non-empty subset sums as a set."""
    n = len(elements)
    sums = set()
    # Iterative subset sum generation
    # Start with {0}
    current_sums = {0}
    for e in elements:
        new_sums = set()
        for s in current_sums:
            new_sums.add(s + e)
        current_sums = current_sums | new_sums
    current_sums.discard(0)
    return current_sums

def _is_divisibility_free(sums_set):
    """Check if no two distinct sums in sums_set divide each other."""
    if not sums_set:
        return True
    sorted_sums = sorted(sums_set)
    max_sum = sorted_sums[-1]
    for s in sorted_sums:
        if s == 0:
            continue
        # Check multiples of s
        # We only need to check if s*k is in the set for k >= 2
        # Since s is the smallest so far in the sorted loop? No, s iterates through all.
        # If we iterate s from smallest to largest, and we find s*k in the set, 
        # and s*k != s, then it's a conflict.
        # Note: if s*k is in the set, and s*k > s, then s divides s*k.
        # We must ensure s != s*k.
        k = 2
        while s * k <= max_sum:
            if s * k in sums_set:
                return False
            k += 1
    return True

def _is_valid(elements):
    """Check if the set satisfies all constraints."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not (1 <= x <= 256):
            return False
    sums_set = _subset_sums_set(elements)
    return _is_divisibility_free(sums_set)

def _check_add(elements, candidate, sums_set):
    """Check if adding candidate maintains validity. Returns new_sums_set if valid, None otherwise."""
    if candidate in elements:
        return None
    
    # Generate new sums efficiently
    new_sums_set = set(sums_set)
    for s in sums_set:
        new_sums_set.add(s + candidate)
    new_sums_set.add(candidate)
    
    if not _is_divisibility_free(new_sums_set):
        return None
    return new_sums_set

def _greedy_grow(elements, candidates, deadline):
    """Greedily grow the set by trying candidates in order."""
    elements = list(elements)
    sums_set = _subset_sums_set(elements)
    used = set(elements)
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        if candidate in used:
            continue
        new_sums = _check_add(elements, candidate, sums_set)
        if new_sums is not None:
            elements.append(candidate)
            used.add(candidate)
            sums_set = new_sums
    
    return elements

def _local_search_improve(elements, all_candidates, deadline):
    """Try to improve the set by swapping elements to enable adding more."""
    elements = list(elements)
    best_size = len(elements)
    best_elements = list(elements)
    
    used = set(elements)
    remaining = [x for x in all_candidates if x not in used]
    
    # Try each element for removal and see if we can add more
    # We sort remaining in descending order to prefer large numbers
    remaining.sort(reverse=True)
    
    for i in range(len(elements)):
        if time.time() > deadline:
            break
        # Remove element i
        reduced = elements[:i] + elements[i+1:]
        
        # Try to grow the reduced set with remaining candidates
        # We use a greedy grow here
        grown = _greedy_grow(reduced, remaining, deadline)
        
        if len(grown) > best_size:
            best_size = len(grown)
            best_elements = list(grown)
            
    return best_elements

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_result = []
    all_candidates = list(range(1, 257))
    
    random.seed(42)
    
    # Strategy 1: Greedy with descending order (prefer large numbers)
    # Large numbers have fewer multiples, so less likely to create divisibility conflicts.
    candidates_desc = sorted(all_candidates, reverse=True)
    result = _greedy_grow([], candidates_desc, deadline)
    if len(result) > len(best_result):
        best_result = result
        
    # Strategy 2: Greedy with ascending order (prefer small numbers, might pack more?)
    candidates_asc = sorted(all_candidates)
    result = _greedy_grow([], candidates_asc, deadline)
    if len(result) > len(best_result):
        best_result = result
        
    # Strategy 3: Randomized Greedy with Local Search
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Random shuffle
        shuffled = all_candidates.copy()
        random.shuffle(shuffled)
        
        # Start with a random small base? Or just empty?
        # Let's try starting with a few random large numbers
        base_size = random.randint(0, 3)
        base = random.sample(all_candidates, base_size)
        base.sort(reverse=True)
        
        if not _is_valid(base):
            continue
            
        # Grow from base
        result = _greedy_grow(base, shuffled, deadline)
        if len(result) > len(best_result):
            best_result = result
            
        # Local search on the result
        if time.time() < deadline - 5:
            improved = _local_search_improve(result, all_candidates, deadline)
            if len(improved) > len(best_result):
                best_result = improved
                
    # Final validation
    if _is_valid(best_result):
        return [int(x) for x in best_result]
    
    # Fallback: Try a few hardcoded large sets if search fails
    # Example: 250, 251, 252, 253, 254, 255, 256
    fallback = [250, 251, 252, 253, 254, 255, 256]
    if _is_valid(fallback):
        return fallback
        
    return [31, 32, 34]

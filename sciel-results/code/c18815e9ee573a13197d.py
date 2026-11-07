import time
import random
from itertools import combinations

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

def _is_valid(elements):
    """Check if the set satisfies all constraints."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not (1 <= x <= 256):
            return False
    sums = _get_sums(elements)
    sums_list = sorted(sums)
    for i in range(len(sums_list)):
        for j in range(i + 1, len(sums_list)):
            if sums_list[j] % sums_list[i] == 0:
                return False
    return True

def _check_addition(current_elements, candidate, current_sums):
    """Check if adding candidate maintains validity. Returns new sums if valid, None otherwise."""
    new_elements = current_elements + [candidate]
    new_sums = set(current_sums)
    for s in current_sums:
        new_sums.add(s + candidate)
    new_sums.add(candidate)
    
    # Check no two distinct sums divide each other
    sums_list = sorted(new_sums)
    for i in range(len(sums_list)):
        a = sums_list[i]
        for j in range(i + 1, len(sums_list)):
            b = sums_list[j]
            if b % a == 0:
                return None
    return new_sums

def _greedy_grow(start_elements, candidates, current_sums):
    """Greedily grow the set by adding candidates that maintain validity."""
    elements = list(start_elements)
    sums = set(current_sums)
    
    for candidate in candidates:
        if candidate in elements:
            continue
        if not (1 <= candidate <= 256):
            continue
        new_sums = _check_addition(elements, candidate, sums)
        if new_sums is not None:
            elements.append(candidate)
            sums = new_sums
    
    return elements, sums

def _try_strategy(start_elements, candidate_order, deadline):
    """Try a construction strategy."""
    if time.time() > deadline:
        return list(start_elements)
    
    elements = list(start_elements)
    sums = _get_sums(elements)
    
    for candidate in candidate_order:
        if time.time() > deadline:
            break
        if candidate in elements:
            continue
        new_sums = _check_addition(elements, candidate, sums)
        if new_sums is not None:
            elements.append(candidate)
            sums = new_sums
    
    return elements

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    best_result = [31, 32, 34]
    
    all_candidates = list(range(1, 257))
    
    # Strategy 1: Start with large numbers, try to add more
    # Large numbers have fewer divisibility issues
    strategies = []
    
    # Start with various large starting sets
    for base_start in range(150, 257, 10):
        strategies.append([base_start])
    for base_start in range(120, 200, 15):
        strategies.append([base_start])
    
    # Also try starting with pairs
    for a in range(150, 257, 20):
        for b in range(a + 1, 257, 20):
            strategies.append([a, b])
    
    random.seed(42)
    
    for strategy_start in strategies:
        if time.time() > deadline:
            break
        
        # Try different orderings of remaining candidates
        remaining = [x for x in all_candidates if x not in strategy_start]
        
        # Try descending order (large to small)
        order1 = sorted(remaining, reverse=True)
        result = _try_strategy(strategy_start, order1, deadline)
        if len(result) > len(best_result):
            best_result = result
        
        if time.time() > deadline:
            break
        
        # Try ascending order
        order2 = sorted(remaining)
        result = _try_strategy(strategy_start, order2, deadline)
        if len(result) > len(best_result):
            best_result = result
        
        if time.time() > deadline:
            break
    
    # Try randomized approaches
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Random start
        start_size = random.randint(1, 3)
        start_elements = random.sample(all_candidates, start_size)
        start_elements.sort()
        
        remaining = [x for x in all_candidates if x not in start_elements]
        random.shuffle(remaining)
        
        result = _try_strategy(start_elements, remaining, deadline)
        if len(result) > len(best_result):
            best_result = result
    
    # Final validation
    if _is_valid(best_result):
        return [int(x) for x in best_result]
    
    # Fallback
    return [31, 32, 34]

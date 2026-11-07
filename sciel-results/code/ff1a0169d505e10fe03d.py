import time
import random
import itertools

def _subset_sums_set(elements):
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

def _is_divisibility_free(sums_set):
    """Check if no two distinct sums in sums_set divide each other."""
    if not sums_set:
        return True
    max_sum = max(sums_set)
    for s in sums_set:
        if s == 0:
            continue
        for k in range(2, max_sum // s + 1):
            if s * k in sums_set:
                return False
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

def _backtrack_search(elements, candidates, depth, best, deadline):
    """Backtracking search to find larger valid sets."""
    if time.time() > deadline:
        return
    if depth > len(best):
        best.extend(elements)
    
    if not candidates:
        return
    
    for i, candidate in enumerate(candidates):
        if time.time() > deadline:
            return
        new_elements = elements + [candidate]
        if _is_valid(new_elements):
            remaining = candidates[i+1:]
            _backtrack_search(new_elements, remaining, depth + 1, best, deadline)

def _local_search_improve(elements, all_candidates, deadline):
    """Try to improve the set by swapping elements to enable adding more."""
    elements = list(elements)
    best_size = len(elements)
    best_elements = list(elements)
    
    used = set(elements)
    remaining = [x for x in all_candidates if x not in used]
    
    # Try each element for removal and see if we can add more
    for i in range(len(elements)):
        if time.time() > deadline:
            break
        # Remove element i
        reduced = elements[:i] + elements[i+1:]
        if not _is_valid(reduced):
            continue
        
        # Try to grow the reduced set
        reduced_used = set(reduced)
        reduced_remaining = [x for x in all_candidates if x not in reduced_used]
        
        # Sort by some heuristic - try larger numbers first
        reduced_remaining.sort(reverse=True)
        
        grown = _greedy_grow(reduced, reduced_remaining, deadline)
        
        if len(grown) > best_size:
            best_size = len(grown)
            best_elements = grown
    
    return best_elements

def _score_candidate(candidate, current_sums):
    """Score a candidate based on how few conflicts it creates."""
    # Count how many multiples of existing sums would be created
    conflicts = 0
    max_sum = max(current_sums) if current_sums else 0
    
    for s in current_sums:
        new_s = s + candidate
        # Check if new_s creates a divisibility conflict
        for k in range(2, (max_sum + candidate) // new_s + 1):
            if new_s * k in current_sums:
                conflicts += 1
                break
        # Also check if new_s is a multiple of some existing sum
        for t in current_sums:
            if t != new_s and new_s % t == 0:
                conflicts += 1
                break
    
    # Lower conflict is better
    return conflicts

def _smart_candidate_order(candidates, current_sums):
    """Order candidates by conflict score (fewest conflicts first)."""
    scored = [( _score_candidate(c, current_sums), c) for c in candidates]
    scored.sort()
    return [c for _, c in scored]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_result = [31, 32, 34]
    all_candidates = list(range(1, 257))
    
    random.seed(42)
    
    # Strategy 1: Backtracking search with smart ordering
    # Start with small bases and try to grow via backtracking
    for start_size in range(1, 4):
        if time.time() > deadline:
            break
        
        for _ in range(30):
            if time.time() > deadline:
                break
            
            start_elements = random.sample(all_candidates, start_size)
            start_elements.sort()
            
            if not _is_valid(start_elements):
                continue
            
            # Use smart ordering for remaining candidates
            used = set(start_elements)
            remaining = [x for x in all_candidates if x not in used]
            
            # Order by conflict score
            sums_set = _subset_sums_set(start_elements)
            ordered = _smart_candidate_order(remaining, sums_set)
            
            # Try greedy first
            result = _greedy_grow(start_elements, ordered, deadline)
            if len(result) > len(best_result):
                best_result = result
            
            # If we have time, try local search
            if time.time() < deadline - 5:
                improved = _local_search_improve(result, all_candidates, deadline)
                if len(improved) > len(best_result):
                    best_result = improved
    
    # Strategy 2: Greedy with multiple random orderings
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Start with a random valid base of size 2-3
        base_size = random.choice([2, 3])
        base = random.sample(all_candidates, base_size)
        base.sort()
        
        if not _is_valid(base):
            continue
        
        used = set(base)
        remaining = [x for x in all_candidates if x not in used]
        
        # Random shuffle
        random.shuffle(remaining)
        
        result = _greedy_grow(base, remaining, deadline)
        if len(result) > len(best_result):
            best_result = result
    
    # Strategy 3: Local search on the best result found so far
    if time.time() < deadline - 3:
        improved = _local_search_improve(best_result, all_candidates, deadline)
        if len(improved) > len(best_result):
            best_result = improved
    
    # Final validation
    if _is_valid(best_result):
        return [int(x) for x in best_result]
    
    # Fallback
    return [31, 32, 34]

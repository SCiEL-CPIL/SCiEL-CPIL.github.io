"""Program using a model-and-optimize approach for large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement
from collections import Counter

def _compute_all_sums(selected):
    """Compute set of all unordered triple sums (with replacement)."""
    n = len(selected)
    seen = set()
    for a in range(n):
        sa = selected[a]
        for b in range(a, n):
            sb = selected[b]
            sa_b = sa + sb
            for c in range(b, n):
                s = sa_b + selected[c]
                if s in seen:
                    return None  # collision detected
                seen.add(s)
    return seen

def _check_valid(selected):
    """Return True if all unordered triple sums (with replacement) are unique."""
    return _compute_all_sums(selected) is not None

def _incremental_check_add(selected, x, existing_sums):
    """Check if adding x to selected keeps all triple sums unique.
    Returns (True, updated_sums) if valid, (False, None) otherwise."""
    n = len(selected)
    new_sums = set()
    # New triples involving x: (x,x,x), (a,x,x) for a in selected, (a,b,x) for a<=b in selected
    # Since x > all elements in selected (we add in sorted order), x is the largest.
    # Triples with x: 
    #   (x,x,x): sum = 3*x
    #   (a,x,x) for a in selected: sum = a + 2*x
    #   (a,b,x) for a<=b in selected: sum = a + b + x
    # Also need to check these don't collide with existing_sums or each other.
    
    # Check (x,x,x)
    s = 3 * x
    if s in existing_sums or s in new_sums:
        return (False, None)
    new_sums.add(s)
    
    # Check (a,x,x) for each a in selected
    for a in selected:
        s = a + 2 * x
        if s in existing_sums or s in new_sums:
            return (False, None)
        new_sums.add(s)
    
    # Check (a,b,x) for a<=b in selected
    for i in range(n):
        sa = selected[i]
        for j in range(i, n):
            sb = selected[j]
            s = sa + sb + x
            if s in existing_sums or s in new_sums:
                return (False, None)
            new_sums.add(s)
    
    # Merge: return combined set
    combined = existing_sums | new_sums
    return (True, combined)

def _greedy_incremental(start_list, limit, deadline):
    """Greedy construction with incremental sum tracking."""
    base = sorted(start_list)
    if not base:
        base = [1]
    
    # Compute initial sums
    sums = _compute_all_sums(base)
    if sums is None:
        return base
    
    while len(base) < 80:
        if time.time() > deadline:
            break
        next_min = max(base) + 1
        found = False
        for x in range(next_min, limit + 1):
            ok, new_sums = _incremental_check_add(base, x, sums)
            if ok:
                base.append(x)
                sums = new_sums
                found = True
                break
        if not found:
            break
    return base

def _local_search_swap(base, limit, deadline, max_iter=200):
    """Try to improve by swapping elements to allow more additions."""
    base = sorted(base)
    if not base:
        return base
    
    sums = _compute_all_sums(base)
    if sums is None:
        return base
    
    best = list(base)
    best_size = len(base)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try removing one element and adding two
        improved = False
        for remove_idx in range(len(base)):
            if time.time() > deadline:
                break
            candidate = [x for i, x in enumerate(base) if i != remove_idx]
            removed = base[remove_idx]
            
            # Recompute sums without removed element
            new_sums = _compute_all_sums(candidate)
            if new_sums is None:
                continue
            
            # Try to add two new elements
            next_min = max(candidate) + 1 if candidate else 1
            added_count = 0
            temp_base = list(candidate)
            temp_sums = new_sums
            
            for x in range(next_min, limit + 1):
                if added_count >= 2:
                    break
                ok, updated = _incremental_check_add(temp_base, x, temp_sums)
                if ok:
                    temp_base.append(x)
                    temp_sums = updated
                    added_count += 1
            
            if len(temp_base) > best_size:
                best = temp_base
                best_size = len(best)
                base = temp_base
                sums = temp_sums
                improved = True
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s margin (90s kill)
    
    limit = 4096
    max_size = 80
    
    best_result = []
    
    # Phase 1: Deterministic greedy from 1 with incremental check
    result = _greedy_incremental([1], limit, deadline)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 2: Local search to try to improve
    result = _local_search_swap(best_result, limit, deadline, max_iter=100)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 3: Try different starting points
    for seed in range(2, 30):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        # Try small random starting sets
        k = random.randint(2, 6)
        initial_set = sorted(random.sample(range(1, min(100, limit+1)), k))
        
        if _check_valid(initial_set):
            result = _greedy_incremental(initial_set, limit, deadline)
            if len(result) > len(best_result):
                best_result = result
                # Try local search on the new best
                if time.time() < deadline - 5:
                    result2 = _local_search_swap(best_result, limit, deadline, max_iter=50)
                    if len(result2) > len(best_result):
                        best_result = result2
    
    return best_result

# EVOLVE-BLOCK-END

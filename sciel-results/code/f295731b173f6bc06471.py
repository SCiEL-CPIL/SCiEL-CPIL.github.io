"""Optimized construction for many 3-term APs without 4-term APs in [1,256]."""
import time
import random
import numpy as np

def _count_3aps(a, n=256):
    """Count distinct 3-term APs (positive difference) in set a."""
    s = set(a)
    cnt = 0
    for x in a:
        for y in a:
            if y > x:
                d = y - x
                if x - d >= 1 and (x - d) in s:
                    cnt += 1
    return cnt

def _has_4ap(a, n=256):
    """Check if set a has a nonconstant 4-term AP."""
    s = set(a)
    for x in a:
        max_d = min(x - 1, (n - x) // 3)
        for d in range(1, max_d + 1):
            if x + d in s and x + 2*d in s and x + 3*d in s:
                return True
    return False

def _evaluate(a):
    """Return (valid, score) where score is number of 3-term APs."""
    if not a:
        return False, 0
    if _has_4ap(a):
        return False, 0
    return True, _count_3aps(a)

def _greedy_construct(n=256):
    """Greedy: add each x if it doesn't create a 4-AP, preferring those that maximize 3-APs."""
    selected = set()
    # Precompute for each x, the 4-APs that would be formed if x is added
    # For efficiency, we just check incrementally
    for x in range(1, n+1):
        # Check if adding x creates a 4-AP
        # A 4-AP ending at x: x-3d, x-2d, x-d, x for d>=1
        # A 4-AP starting at x: x, x+d, x+2d, x+3d for d>=1
        # A 4-AP with x as second: x-d, x, x+d, x+2d
        # A 4-AP with x as third: x-2d, x-d, x, x+d
        creates_4ap = False
        # Check as first element
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
                creates_4ap = True
                break
        if not creates_4ap:
            # Check as second element
            max_d2 = min(x - 1, (n - x) // 2)
            for d in range(1, max_d2 + 1):
                if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
                    creates_4ap = True
                    break
        if not creates_4ap:
            # Check as third element
            max_d3 = min((x - 1) // 2, n - x)
            for d in range(1, max_d3 + 1):
                if (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
                    creates_4ap = True
                    break
        if not creates_4ap:
            # Check as fourth element
            max_d4 = (x - 1) // 3
            for d in range(1, max_d4 + 1):
                if (x - d) in selected and (x - 2*d) in selected and (x - 3*d) in selected:
                    creates_4ap = True
                    break
        if not creates_4ap:
            selected.add(x)
    return sorted(selected)

def _local_search(a, n=256, max_iter=5000, time_limit=50.0):
    """Local search to improve the number of 3-term APs."""
    a_set = set(a)
    current_score = _count_3aps(a)
    best_a = list(a)
    best_score = current_score
    start_time = time.time()
    
    # Precompute all elements
    all_elems = list(range(1, n+1))
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Try swaps: remove one element, add another
        # Or just try adding/removing
        # Let's try: for each element not in a, try adding it
        # For each element in a, try removing it
        
        # First, try adding elements
        best_gain = 0
        best_add = None
        for x in all_elems:
            if x in a_set:
                continue
            # Check if adding x creates a 4-AP
            creates_4ap = False
            max_d = (n - x) // 3
            for d in range(1, max_d + 1):
                if (x + d) in a_set and (x + 2*d) in a_set and (x + 3*d) in a_set:
                    creates_4ap = True
                    break
            if not creates_4ap:
                max_d2 = min(x - 1, (n - x) // 2)
                for d in range(1, max_d2 + 1):
                    if (x - d) in a_set and (x + d) in a_set and (x + 2*d) in a_set:
                        creates_4ap = True
                        break
            if not creates_4ap:
                max_d3 = min((x - 1) // 2, n - x)
                for d in range(1, max_d3 + 1):
                    if (x - 2*d) in a_set and (x - d) in a_set and (x + d) in a_set:
                        creates_4ap = True
                        break
            if not creates_4ap:
                max_d4 = (x - 1) // 3
                for d in range(1, max_d4 + 1):
                    if (x - d) in a_set and (x - 2*d) in a_set and (x - 3*d) in a_set:
                        creates_4ap = True
                        break
            if not creates_4ap:
                # Count new 3-APs
                new_3aps = 0
                for y in a_set:
                    if y > x:
                        d = y - x
                        if x - d >= 1 and (x - d) in a_set:
                            new_3aps += 1
                    else:
                        d = x - y
                        if x + d <= n and (x + d) in a_set:
                            new_3aps += 1
                if new_3aps > best_gain:
                    best_gain = new_3aps
                    best_add = x
        
        if best_add is not None:
            a_set.add(best_add)
            current_score += best_gain
            if current_score > best_score:
                best_score = current_score
                best_a = list(a_set)
            continue
        
        # Try removing elements
        best_loss = 0
        best_remove = None
        for x in list(a_set):
            # Count 3-APs that x participates in
            loss = 0
            for y in a_set:
                if y > x:
                    d = y - x
                    if x - d >= 1 and (x - d) in a_set:
                        loss += 1
                else:
                    d = x - y
                    if x + d <= n and (x + d) in a_set:
                        loss += 1
            if loss < best_loss:
                best_loss = loss
                best_remove = x
        
        if best_remove is not None and best_loss < best_gain or (best_add is None and best_loss < 0):
            # Only remove if it doesn't hurt (or helps by enabling better additions)
            # For now, only remove if loss is negative (which can't happen) or if we're stuck
            pass
        
        # Try swaps
        best_swap_gain = 0
        best_swap = None
        for x_remove in list(a_set):
            # Compute loss of removing x_remove
            loss = 0
            for y in a_set:
                if y == x_remove:
                    continue
                if y > x_remove:
                    d = y - x_remove
                    if x_remove - d >= 1 and (x_remove - d) in a_set:
                        loss += 1
                else:
                    d = x_remove - y
                    if x_remove + d <= n and (x_remove + d) in a_set:
                        loss += 1
            
            for x_add in all_elems:
                if x_add in a_set:
                    continue
                # Check if adding x_add creates 4-AP (after removing x_remove)
                temp_set = a_set - {x_remove}
                creates_4ap = False
                max_d = (n - x_add) // 3
                for d in range(1, max_d + 1):
                    if (x_add + d) in temp_set and (x_add + 2*d) in temp_set and (x_add + 3*d) in temp_set:
                        creates_4ap = True
                        break
                if not creates_4ap:
                    max_d2 = min(x_add - 1, (n - x_add) // 2)
                    for d in range(1, max_d2 + 1):
                        if (x_add - d) in temp_set and (x_add + d) in temp_set and (x_add + 2*d) in temp_set:
                            creates_4ap = True
                            break
                if not creates_4ap:
                    max_d3 = min((x_add - 1) // 2, n - x_add)
                    for d in range(1, max_d3 + 1):
                        if (x_add - 2*d) in temp_set and (x_add - d) in temp_set and (x_add + d) in temp_set:
                            creates_4ap = True
                            break
                if not creates_4ap:
                    max_d4 = (x_add - 1) // 3
                    for d in range(1, max_d4 + 1):
                        if (x_add - d) in temp_set and (x_add - 2*d) in temp_set and (x_add - 3*d) in temp_set:
                            creates_4ap = True
                            break
                if not creates_4ap:
                    gain = 0
                    for y in temp_set:
                        if y > x_add:
                            d = y - x_add
                            if x_add - d >= 1 and (x_add - d) in temp_set:
                                gain += 1
                        else:
                            d = x_add - y
                            if x_add + d <= n and (x_add + d) in temp_set:
                                gain += 1
                    net = gain - loss
                    if net > best_swap_gain:
                        best_swap_gain = net
                        best_swap = (x_remove, x_add)
        
        if best_swap is not None and best_swap_gain > 0:
            x_remove, x_add = best_swap
            a_set.remove(x_remove)
            a_set.add(x_add)
            current_score += best_swap_gain
            if current_score > best_score:
                best_score = current_score
                best_a = list(a_set)
        else:
            # No improvement, break
            break
    
    return best_a, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline with 10s margin
    
    # Start with greedy construction
    a = _greedy_construct(256)
    score = _count_3aps(a)
    
    # Local search
    if time.time() < deadline - 5:
        a, score = _local_search(a, n=256, max_iter=10000, time_limit=deadline - time.time() - 5)
    
    return sorted(a)

# EVOLVE-BLOCK-END

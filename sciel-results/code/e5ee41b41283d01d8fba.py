"""Rewritten program for small inclusion-maximal Sidon sets in [1,256]."""
import time
import random
from itertools import combinations

N = 256
TIME_LIMIT = 55.0  # safety margin before 60s kill

def _get_sums(selected):
    """Compute all pair sums a+b for a<=b in selected."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a Sidon set (all a+b for a<=b distinct)."""
    n = len(selected)
    expected = n * (n + 1) // 2
    sums = _get_sums(selected)
    return len(sums) == expected

def _can_add(selected, x, sums):
    """Check if x can be added to selected without breaking Sidon property."""
    new_sums = set()
    for a in selected:
        s = a + x
        if s in sums or s in new_sums:
            return False
        new_sums.add(s)
    s = 2 * x
    if s in sums or s in new_sums:
        return False
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal in [1,N]."""
    sel_set = set(selected)
    sums = _get_sums(selected)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _can_add(selected, x, sums):
            return False
    return True

def _greedy_construct(order):
    """Greedy construction using given order of elements."""
    selected = []
    sums = set()
    for x in order:
        if _can_add(selected, x, sums):
            selected.append(x)
            for a in selected[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    return selected

def _try_remove(selected, N):
    """Try to remove elements while maintaining Sidon and maximality."""
    selected = list(selected)
    changed = True
    while changed:
        changed = False
        for i in range(len(selected) - 1, -1, -1):
            candidate = selected[i]
            new_sel = selected[:i] + selected[i+1:]
            if _is_sidon(new_sel) and _is_maximal(new_sel, N):
                selected = new_sel
                changed = True
    return selected

def _local_search(selected, N, time_limit):
    """Local search: try removing elements, then targeted swaps."""
    start = time.time()
    selected = list(selected)
    
    # Phase 1: Try removing elements (most impactful for size reduction)
    improved = True
    while improved:
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(len(selected) - 1, -1, -1):
            if time.time() - start > time_limit:
                break
            new_sel = selected[:i] + selected[i+1:]
            if _is_sidon(new_sel) and _is_maximal(new_sel, N):
                selected = new_sel
                improved = True
    
    # Phase 2: Try removing pairs of elements (more aggressive)
    if time.time() - start < time_limit * 0.7:
        improved = True
        while improved:
            if time.time() - start > time_limit:
                break
            improved = False
            for i in range(len(selected) - 1):
                if time.time() - start > time_limit:
                    break
                for j in range(i + 1, len(selected)):
                    if time.time() - start > time_limit:
                        break
                    new_sel = selected[:i] + selected[i+1:j] + selected[j+1:]
                    if _is_sidon(new_sel) and _is_maximal(new_sel, N):
                        selected = new_sel
                        improved = True
                        break
    
    # Phase 3: Targeted swaps - try replacing elements with smaller ones
    if time.time() - start < time_limit * 0.85:
        for _ in range(20):
            if time.time() - start > time_limit:
                break
            sel_set = set(selected)
            best_swap = None
            for i in range(len(selected)):
                if time.time() - start > time_limit:
                    break
                old = selected[i]
                # Try replacing with smaller elements first
                for x in range(1, min(old, N + 1)):
                    if x in sel_set:
                        continue
                    new_sel = selected[:i] + [x] + selected[i+1:]
                    if _is_sidon(new_sel) and _is_maximal(new_sel, N):
                        best_swap = (i, x)
                        break
                if best_swap:
                    break
            if best_swap:
                i, x = best_swap
                selected = selected[:i] + [x] + selected[i+1:]
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best = None
    best_size = float('inf')
    
    # Try multiple random orderings with greedy construction
    rng = random.Random(42)
    
    # First try the natural ascending order (baseline)
    asc_order = list(range(1, N + 1))
    sel = _greedy_construct(asc_order)
    if len(sel) < best_size:
        best = sel
        best_size = len(sel)
    
    # Try descending order
    desc_order = list(range(N, 0, -1))
    sel = _greedy_construct(desc_order)
    if len(sel) < best_size:
        best = sel
        best_size = len(sel)
    
    # Try random orderings with more attempts since validation is faster
    num_tries = 400
    for attempt in range(num_tries):
        if time.time() > deadline - 8.0:
            break
        
        # Generate a random permutation
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        sel = _greedy_construct(order)
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)
            # Try to improve with local search
            remaining = deadline - time.time() - 3.0
            if remaining > 5.0:
                improved = _local_search(sel, N, remaining)
                if len(improved) < best_size:
                    best = improved
                    best_size = len(best)
    
    # Final polish on best solution
    remaining = deadline - time.time() - 2.0
    if remaining > 2.0 and best is not None:
        polished = _local_search(best, N, remaining)
        if len(polished) < best_size:
            best = polished
            best_size = len(best)
    
    # Verify the result
    if best is None:
        # Fallback: return ascending greedy result
        best = _greedy_construct(list(range(1, N + 1)))
    
    # Final validation
    if not _is_sidon(best):
        # Should not happen, but safety check
        best = _greedy_construct(list(range(1, N + 1)))
    if not _is_maximal(best, N):
        # If not maximal, add elements until maximal
        sel_set = set(best)
        sums = _get_sums(best)
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            if _can_add(best, x, sums):
                best.append(x)
                sel_set.add(x)
                for a in best[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
        best.sort()
    
    return best

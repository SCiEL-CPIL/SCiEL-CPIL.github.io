"""Rewritten program for small inclusion-maximal Sidon sets in [1,256]."""
import time
import random
from itertools import combinations

N = 256
TIME_LIMIT = 55.0  # safety margin before 60s kill

def _is_sidon(selected):
    """Check if selected is a Sidon set (all a+b for a<=b distinct)."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal in [1,N]."""
    sel_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            sums.add(a + b)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x would break Sidon property
        new_sums = set()
        ok = True
        for a in selected:
            s = a + x
            if s in sums or s in new_sums:
                ok = False
                break
            new_sums.add(s)
        if not ok:
            continue
        s = 2 * x
        if s in sums or s in new_sums:
            continue
        # x can be added, so not maximal
        return False
    return True

def _greedy_construct(order):
    """Greedy construction using given order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = []
        for a in selected:
            added.append(x + a)
        added.append(2 * x)
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
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
    """Local search: try removing elements, then swapping."""
    start = time.time()
    selected = list(selected)
    
    # Phase 1: Try removing elements
    improved = True
    while improved:
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(len(selected) - 1, -1, -1):
            if time.time() - start > time_limit:
                break
            candidate = selected[i]
            new_sel = selected[:i] + selected[i+1:]
            if _is_sidon(new_sel) and _is_maximal(new_sel, N):
                selected = new_sel
                improved = True
    
    # Phase 2: Try swapping one element for another (to reduce size or maintain)
    for _ in range(50):  # limited iterations
        if time.time() - start > time_limit:
            break
        best_swap = None
        best_size = len(selected)
        sel_set = set(selected)
        
        for i in range(len(selected)):
            if time.time() - start > time_limit:
                break
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                new_sel = selected[:i] + [x] + selected[i+1:]
                if len(new_sel) < best_size and _is_sidon(new_sel) and _is_maximal(new_sel, N):
                    best_size = len(new_sel)
                    best_swap = (i, x)
                    if best_size == 1:
                        return new_sel
        if best_swap is not None:
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
    
    # Try random orderings
    num_tries = 200
    for attempt in range(num_tries):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a random permutation
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        sel = _greedy_construct(order)
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)
            # Try to improve with local search
            remaining = deadline - time.time() - 2.0
            if remaining > 3.0:
                improved = _local_search(sel, N, remaining)
                if len(improved) < best_size:
                    best = improved
                    best_size = len(best)
    
    # Final polish on best solution
    remaining = deadline - time.time() - 1.0
    if remaining > 1.0 and best is not None:
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
        sums = set()
        for i, a in enumerate(best):
            for j in range(i, len(best)):
                sums.add(a + best[j])
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            added = [x + a for a in best] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                best.append(x)
                sel_set.add(x)
                sums.update(added)
        best.sort()
    
    return best

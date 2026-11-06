import random
import time

N = 256

def _get_sums(sel):
    s = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s.add(a + sel[j])
    return s

def _can_add(sel, sums, x):
    for a in sel:
        s = x + a
        if s in sums:
            return False
    s2 = 2 * x
    if s2 in sums:
        return False
    return True

def _greedy_random(order, deadline):
    sel = []
    sums = set()
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(sel, sums, x):
            sel.append(x)
            for a in sel[:-1]:
                sums.add(x + a)
            sums.add(2 * x)
    return sel

def _saturate(sel, sums, deadline):
    """Add all addable elements in ascending order to make maximal."""
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x in set(sel):
                continue
            if _can_add(sel, sums, x):
                sel.append(x)
                for a in sel[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
                changed = True
    return sel

def _local_search_reduce(sel, deadline, max_iters=200):
    """Try to remove elements and re-saturate to get smaller set."""
    best = list(sel)
    best_sums = _get_sums(best)
    best_size = len(best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Try removing each element
        for i in range(len(best)):
            if time.time() > deadline:
                break
            candidate = best[:i] + best[i+1:]
            cand_sums = _get_sums(candidate)
            # Saturate
            changed = True
            while changed:
                changed = False
                for x in range(1, N + 1):
                    if x in set(candidate):
                        continue
                    if _can_add(candidate, cand_sums, x):
                        candidate.append(x)
                        for a in candidate[:-1]:
                            cand_sums.add(x + a)
                        cand_sums.add(2 * x)
                        changed = True
            if len(candidate) < best_size:
                best = candidate
                best_sums = cand_sums
                best_size = len(best)
    
    return best

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s limit
    
    best_set = None
    best_size = float('inf')
    
    # Phase 1: Try many random orderings
    for trial in range(500):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        sel = _greedy_random(order, deadline)
        # Saturate to make maximal
        sel_sums = _get_sums(sel)
        sel = _saturate(sel, sel_sums, deadline)
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
            # Try local search
            improved = _local_search_reduce(sel, deadline, max_iters=30)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = improved
    
    # Phase 2: More targeted local search on best found
    if best_set is not None and time.time() < deadline - 5:
        improved = _local_search_reduce(best_set, deadline, max_iters=100)
        if len(improved) < best_size:
            best_set = improved
    
    # Fallback: simple greedy if nothing found
    if best_set is None:
        sel = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(sel, sums, x):
                sel.append(x)
                for a in sel[:-1]:
                    sums.add(x + a)
                sums.add(2 * x)
        best_set = sel
    
    return best_set

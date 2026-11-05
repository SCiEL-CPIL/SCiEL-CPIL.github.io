"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_valid(selected):
    """Check no subset of selected sums to 200."""
    reachable = {0}
    for x in selected:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 in new:
            return False
        reachable |= new
    return True

def _greedy(order):
    """Greedy: add elements in given order if valid."""
    selected, reachable = [], {0}
    for x in order:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected

def _try_add(selected, x):
    """Try adding x to selected; return new list if valid, else None."""
    if x in selected:
        return None
    reachable = {0}
    for y in selected:
        new = {s + y for s in reachable if s + y <= 200}
        if 200 in new:
            return None
        reachable |= new
    new = {s + x for s in reachable if s + x <= 200}
    if 200 in new:
        return None
    return selected + [x]

def construct():
    deadline = time.time() + 80  # safety margin
    
    # Initial greedy from largest to smallest
    best = _greedy(range(100, 0, -1))
    best_set = set(best)
    
    # Try to add back excluded elements
    excluded = [x for x in range(1, 101) if x not in best_set]
    improved = True
    while improved:
        improved = False
        for x in excluded[:]:
            if time.time() > deadline:
                break
            result = _try_add(best, x)
            if result is not None:
                best = result
                best_set = set(best)
                excluded.remove(x)
                improved = True
        if time.time() > deadline:
            break
    
    # Try swaps: remove one, add two excluded
    for _ in range(50):  # limited swap iterations
        if time.time() > deadline:
            break
        swapped = False
        for i, rem in enumerate(best):
            if time.time() > deadline:
                break
            candidate = best[:i] + best[i+1:]
            cand_set = set(candidate)
            # Try adding pairs of excluded elements
            ex = [x for x in excluded if x not in cand_set]
            for j in range(len(ex)):
                if time.time() > deadline:
                    break
                r1 = _try_add(candidate, ex[j])
                if r1 is not None:
                    for k in range(len(ex)):
                        if k == j:
                            continue
                        r2 = _try_add(r1, ex[k])
                        if r2 is not None and len(r2) > len(best):
                            best = r2
                            best_set = set(best)
                            excluded = [x for x in range(1, 101) if x not in best_set]
                            swapped = True
                            break
                    if swapped:
                        break
                if swapped:
                    break
            if swapped:
                break
        if not swapped:
            break
    
    # Final verification
    if _is_valid(best) and len(best) > 0:
        return best
    # Fallback to original greedy
    return _greedy(range(100, 0, -1))
# EVOLVE-BLOCK-END

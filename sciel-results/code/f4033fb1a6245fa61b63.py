"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _construct_greedy_desc():
    """Greedy: process 100..1, add x if 200 not reachable via x."""
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected

def _construct_greedy_asc():
    """Greedy: process 1..100, add x if 200 not reachable via x."""
    selected, reachable = [], {0}
    for x in range(1, 101):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected

def _construct_greedy_skip_large():
    """Greedy descending but skip elements > 100 (none here) or use a threshold."""
    # Try skipping some large elements to see if we can fit more small ones
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        if x > 90:  # skip some large ones initially
            continue
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    # Now try to add the skipped large ones
    for x in range(100, 90, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected

def _construct_local_search(start_set):
    """Local search: try to add any excluded element, or swap."""
    selected = set(start_set)
    excluded = set(range(1, 101)) - selected
    
    def _reachable_sums(s):
        r = {0}
        for x in s:
            new = {v + x for v in r if v + x <= 200}
            r |= new
        return r
    
    changed = True
    while changed:
        changed = False
        reachable = _reachable_sums(selected)
        
        # Try adding any excluded element
        for x in sorted(excluded, reverse=True):
            if 200 - x in reachable and x not in selected:
                # Adding x would create 200, so skip
                continue
            # Check if adding x creates 200
            new_reach = set(reachable)
            for s in reachable:
                if s + x <= 200:
                    new_reach.add(s + x)
            if 200 not in new_reach:
                selected.add(x)
                excluded.discard(x)
                changed = True
                break
        
        if changed:
            continue
        
        # Try swaps: remove one from selected, add one from excluded
        for x in sorted(selected, key=lambda v: -v):
            for y in sorted(excluded, key=lambda v: v):
                if y > 100 or y < 1:
                    continue
                new_set = selected - {x} | {y}
                if len(new_set) != len(selected):
                    continue
                r = _reachable_sums(new_set)
                if 200 not in r:
                    selected = new_set
                    excluded = set(range(1, 101)) - selected
                    changed = True
                    break
            if changed:
                break
    
    return list(selected)

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    # Generate multiple candidates
    candidates = []
    
    # Greedy descending (original)
    c1 = _construct_greedy_desc()
    candidates.append(c1)
    
    # Greedy ascending
    c2 = _construct_greedy_asc()
    candidates.append(c2)
    
    # Greedy with skip
    c3 = _construct_greedy_skip_large()
    candidates.append(c3)
    
    # Find best by cardinality
    best = max(candidates, key=len)
    
    # Local search to improve
    if time.time() < deadline - 5:
        improved = _construct_local_search(best)
        if len(improved) > len(best):
            best = improved
    
    # Verify the result is valid
    reachable = {0}
    for x in best:
        new = {s + x for s in reachable if s + x <= 200}
        reachable |= new
    if 200 in reachable:
        # Fallback to the greedy descending which is known to work
        return c1
    
    return best
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_DEADLINE = 150.0
_MARGIN = 5.0

def _valid(s):
    """Sidon check: all positive pairwise differences distinct; covers 1..30; has 0."""
    if 0 not in s:
        return False
    n = len(s)
    if n > 32:
        return False
    if max(s) > 10**6:
        return False
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d <= 0:
                return False
            if d in seen:
                return False
            seen.add(d)
    for d in range(1, 31):
        if d not in seen:
            return False
    return True

def _is_sidon(elements, seen_diffs, x):
    """Check if adding x to elements (sorted, all < x) maintains Sidon property.
    Returns (valid, new_diffs) where new_diffs are the differences x-e for e in elements."""
    new_diffs = []
    for e in elements:
        d = x - e
        if d in seen_diffs:
            return False, None
        new_diffs.append(d)
    # new_diffs are all distinct by construction since elements are distinct
    return True, new_diffs

def _dfs(elements, seen_diffs, covered_diffs, best_max_elem, deadline, max_size=32):
    """DFS to find a Sidon set covering 1..30 with max element < best_max_elem.
    elements: sorted list, starts with [0]
    seen_diffs: set of all differences so far
    covered_diffs: set of differences in 1..30 that are in seen_diffs
    Returns (result, new_best_max) or (None, best_max_elem)"""
    target_diffs = frozenset(range(1, 31))
    
    if time.time() > deadline:
        return None, best_max_elem
    
    if covered_diffs == target_diffs:
        if len(elements) <= max_size:
            m = elements[-1]
            if m < best_max_elem:
                return list(elements), m
        return None, best_max_elem
    
    if len(elements) >= max_size:
        return None, best_max_elem
    
    remaining = target_diffs - covered_diffs
    if not remaining:
        return None, best_max_elem
    
    current_max = elements[-1]
    upper_bound = best_max_elem - 1
    
    # Generate candidates: x = e + d for e in elements, d in remaining
    # Also try x = current_max + 1 (small step) as a fallback
    candidates = set()
    for e in elements:
        for d in remaining:
            x = e + d
            if x > current_max and x <= upper_bound:
                candidates.add(x)
    
    # Sort candidates by value (prefer smaller for tighter packing)
    for x in sorted(candidates):
        if time.time() > deadline:
            return None, best_max_elem
        
        valid, new_diffs = _is_sidon(elements, seen_diffs, x)
        if not valid:
            continue
        
        new_seen = seen_diffs | set(new_diffs)
        new_covered = covered_diffs | (set(new_diffs) & target_diffs)
        
        # Pruning: check if we can still cover remaining diffs
        # With k more elements to add, each adds at most len(elements)+1 new diffs
        # but many will be > 30 or duplicates
        # Simple bound: remaining diffs that can be formed
        rem = target_diffs - new_covered
        if rem:
            # The max remaining diff must be formable
            # x_max possible = upper_bound, so max new diff = upper_bound - elements[0] = upper_bound
            # If max(rem) > upper_bound, we can't cover it
            if max(rem) > upper_bound:
                continue
        
        result, new_best = _dfs(
            elements + [x],
            new_seen,
            new_covered,
            new_best if new_best < best_max_elem else best_max_elem,
            deadline,
            max_size
        )
        if result is not None:
            return result, new_best
    
    return None, best_max_elem

def _dfs_with_bound(max_elem_bound, deadline):
    """Run DFS with a hard upper bound on the max element.
    Returns (result, max_elem) or (None, inf)"""
    target_diffs = frozenset(range(1, 31))
    best_max = max_elem_bound + 1  # exclusive upper bound for search
    
    result, new_best = _dfs([0], set(), set(), best_max, deadline)
    if result is not None:
        return result, new_best
    return None, float('inf')

def _local_optimize(s, deadline):
    """Local optimization: try to reduce max element by replacing elements.
    Uses simulated-annealing-like approach with deterministic + random moves."""
    import random
    rng = random.Random(12345)
    
    s = sorted(set(s))
    best_s = list(s)
    best_max = max(s)
    
    for iteration in range(5000):
        if time.time() > deadline - _MARGIN:
            break
        
        # Try each element (except 0) for replacement
        idx = rng.randint(1, len(s) - 1)
        old = s[idx]
        
        # Try a range of new values
        found = False
        for _ in range(20):
            if time.time() > deadline - _MARGIN:
                break
            new = rng.randint(1, best_max - 1)
            if new in s:
                continue
            
            trial = sorted([x for x in s if x != old] + [new])
            if _valid(trial):
                m = max(trial)
                if m < best_max:
                    s = trial
                    best_s = list(s)
                    best_max = m
                    found = True
                    break
        
        # Also try removing an element if still valid
        if not found and len(s) > 9:
            idx2 = rng.randint(1, len(s) - 1)
            trial = [x for x in s if x != s[idx2]]
            if _valid(trial):
                s = trial
                best_s = list(s)
                best_max = max(s)
    
    return best_s, best_max

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    
    # Phase 1: Deterministic DFS to find a good initial solution
    # Start with a generous bound and tighten
    initial_bound = 200
    best_result = None
    best_max = float('inf')
    
    # Try decreasing bounds
    for bound in [200, 150, 120, 100, 80, 70, 60, 50, 45, 40, 38, 36, 34, 32]:
        if time.time() > deadline - 30:
            break
        
        result, m = _dfs_with_bound(bound, deadline)
        if result is not None:
            best_result = result
            best_max = m
            # Continue trying to tighten
            # Don't break; try smaller bounds
        else:
            # No solution at this bound, try slightly larger
            # Actually, if no solution at bound B, there might not be one at smaller bounds
            # But there could be at larger bounds. Since we're going down, if we fail at B,
            # we've already tried larger. So we can stop.
            # Wait, we're trying from large to small. If we fail at B, smaller bounds are harder.
            # But we might have found something at a larger bound already.
            # Let's not break, just continue to try smaller in case of search incompleteness
            pass
    
    # Phase 2: If we have a result, do local optimization
    if best_result is not None:
        optimized, opt_max = _local_optimize(best_result, deadline)
        if opt_max < best_max:
            best_result = optimized
            best_max = opt_max
    
    # Phase 3: If no result yet, try a greedy construction as fallback
    if best_result is None:
        # Greedy: build set by adding elements that cover new differences
        a = [0]
        seen = set()
        
        def _get_diffs(ss):
            diffs = []
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    diffs.append(ss[j] - ss[i])
            return diffs
        
        for d in range(1, 31):
            if d in seen:
                continue
            placed = False
            # Try to find an element x such that x - e = d for some e in a,
            # or e - x = d (but x < e, so x = e - d)
            candidates = set()
            for e in a:
                x1 = e + d
                x2 = e - d
                if x1 >= 0:
                    candidates.add(x1)
                if x2 >= 0:
                    candidates.add(x2)
            
            for x in sorted(candidates):
                if x in a:
                    continue
                trial = sorted(a + [x])
                trial_diffs = _get_diffs(trial)
                if len(set(trial_diffs)) == len(trial_diffs):
                    a = trial
                    seen = set(trial_diffs)
                    placed = True
                    break
            
            if not placed:
                # Add a new element far away
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    trial_diffs = _get_diffs(trial)
                    if len(set(trial_diffs)) == len(trial_diffs):
                        a = trial
                        seen = set(trial_diffs)
                        break
                    y += 1
        
        if _valid(a):
            best_result = a
            best_max = max(a)
            
            # Local optimize
            optimized, opt_max = _local_optimize(best_result, deadline)
            if opt_max < best_max:
                best_result = optimized
                best_max = opt_max
    
    # Phase 4: Final attempt with DFS using the current best as bound
    if time.time() < deadline - _MARGIN:
        if best_result is not None:
            # Try to improve with DFS using current best_max - 1 as bound
            result, m = _dfs_with_bound(best_max - 1, deadline)
            if result is not None and m < best_max:
                best_result = result
                best_max = m
        else:
            # No result yet, try DFS with a large bound
            result, m = _dfs_with_bound(500, deadline)
            if result is not None:
                best_result = result
                best_max = m
    
    if best_result is None:
        # Ultimate fallback
        a = [0]
        def diffs(ss):
            return [y - x for i, x in enumerate(ss) for y in ss[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            placed = False
            for x in sorted(set(v + dd for v in a for dd in (d, -d) if v + dd >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    placed = True
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
    
    return best_result
# EVOLVE-BLOCK-END

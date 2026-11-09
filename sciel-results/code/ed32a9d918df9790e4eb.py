"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _find_violations_detailed(selected):
    """
    Find all violations in the selected set.
    Returns (violations, divisor_counts, summand_counts) where:
    - violations: list of (a, b, c, d) where a divides b+c+d
    - divisor_counts: dict mapping element -> count of violations where it's the divisor
    - summand_counts: dict mapping element -> count of violations where it's a summand
    """
    n = len(selected)
    if n < 4:
        return [], {}, {}
    
    violations = []
    divisor_counts = {x: 0 for x in selected}
    summand_counts = {x: 0 for x in selected}
    
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            s = t[0] + t[1] + t[2]
            if s % a == 0:
                violations.append((a, t[0], t[1], t[2]))
                divisor_counts[a] += 1
                summand_counts[t[0]] += 1
                summand_counts[t[1]] += 1
                summand_counts[t[2]] += 1
                
    return violations, divisor_counts, summand_counts

def _greedy_removal(initial_set, time_limit=10.0):
    """
    Start with initial_set and greedily remove elements to satisfy constraints.
    Prefer removing elements that are divisors in violations.
    """
    selected = list(initial_set)
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        violations, divisor_counts, summand_counts = _find_violations_detailed(selected)
        
        if not violations:
            return selected
            
        # Score each element: prefer removing those with high divisor count
        max_score = -1
        to_remove = None
        for x in selected:
            dc = divisor_counts.get(x, 0)
            sc = summand_counts.get(x, 0)
            score = dc * 2 + sc
            if score > max_score:
                max_score = score
                to_remove = x
            elif score == max_score:
                if x < to_remove:
                    to_remove = x
        
        if to_remove is not None:
            selected.remove(to_remove)
        
    return selected

def _try_swap_back(selected, removed, time_limit=5.0):
    """
    Try to swap removed elements back into the set to increase cardinality.
    """
    selected = list(selected)
    start_time = time.time()
    
    for elem in sorted(removed, reverse=True):
        if time.time() - start_time > time_limit:
            break
        if elem in selected:
            continue
        test_set = selected + [elem]
        violations, _, _ = _find_violations_detailed(test_set)
        if not violations:
            selected = test_set
            removed.remove(elem)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    best = []
    
    pools = [
        list(range(144, 193)),
        list(range(143, 193)),
        list(range(142, 193)),
        list(range(145, 193)),
        list(range(146, 193)),
    ]
    
    for pool in pools:
        if time.time() >= deadline:
            break
        
        candidate = _greedy_removal(pool, time_limit=5.0)
        
        if time.time() < deadline:
            removed = [x for x in range(1, 193) if x not in candidate]
            candidate = _try_swap_back(candidate, removed, time_limit=3.0)
        
        if len(candidate) > len(best):
            best = candidate
        
        if len(best) >= 48:
            break
    
    if best:
        violations, _, _ = _find_violations_detailed(best)
        if violations:
            best = [189, 190, 191, 192]
    
    if not best:
        best = [189, 190, 191, 192]
    
    return best

# EVOLVE-BLOCK-END

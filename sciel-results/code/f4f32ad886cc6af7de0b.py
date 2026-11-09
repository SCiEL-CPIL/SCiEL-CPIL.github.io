"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_validity(selected):
    """Check if the set is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    if n < 4:
        return True
    sel = list(selected)
    for i, a in enumerate(sel):
        others = [sel[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _count_violations_for_new(selected, x):
    """Count violations if we add x to selected. Returns 0 if valid."""
    if x in selected:
        return 1
    n = len(selected)
    violations = 0
    
    # Check x as divisor: does x divide sum of any 3 from selected?
    for triple in combinations(selected, 3):
        if sum(triple) % x == 0:
            violations += 1
    
    # Check each existing a: does a divide sum of any triple that includes x?
    for i, a in enumerate(selected):
        rest = [selected[j] for j in range(n) if j != i]
        for pair in combinations(rest, 2):
            s = x + pair[0] + pair[1]
            if s % a == 0:
                violations += 1
    
    return violations

def _greedy_with_priority(deadline):
    """Greedy that picks elements with fewest violations first."""
    selected = []
    selected_set = set()
    available = set(range(1, 193))
    
    while len(selected) < 48 and time.time() < deadline:
        best_x = None
        best_violations = float('inf')
        
        for x in sorted(available, reverse=True):  # Prefer larger numbers
            v = _count_violations_for_new(selected, x)
            if v == 0:
                best_x = x
                best_violations = 0
                break
            elif v < best_violations:
                best_violations = v
                best_x = x
        
        if best_x is None or best_violations > 0:
            # No valid addition, try with fewer elements or break
            break
        
        selected.append(best_x)
        selected_set.add(best_x)
        available.discard(best_x)
    
    return selected

def _greedy_largest_first(deadline):
    """Simple greedy: try largest numbers first."""
    selected = []
    selected_set = set()
    
    for x in range(192, 0, -1):
        if len(selected) >= 48 or time.time() >= deadline:
            break
        if x in selected_set:
            continue
        
        v = _count_violations_for_new(selected, x)
        if v == 0:
            selected.append(x)
            selected_set.add(x)
    
    return selected

def _local_search(selected, deadline, max_iter=100):
    """Try to improve by swapping elements to increase size or maintain validity."""
    selected = list(selected)
    selected_set = set(selected)
    available = [x for x in range(1, 193) if x not in selected_set]
    
    improved = True
    iterations = 0
    while improved and iterations < max_iter and time.time() < deadline:
        improved = False
        iterations += 1
        
        # Try adding one more element
        for x in sorted(available, reverse=True):
            v = _count_violations_for_new(selected, x)
            if v == 0:
                selected.append(x)
                selected_set.add(x)
                available.remove(x)
                improved = True
                break
        
        # Try swapping: remove one, add another
        if not improved:
            for i in range(len(selected)):
                if time.time() >= deadline:
                    break
                removed = selected[i]
                temp_selected = selected[:i] + selected[i+1:]
                temp_set = set(temp_selected)
                
                for x in sorted(available, reverse=True):
                    if x in temp_set:
                        continue
                    v = _count_violations_for_new(temp_selected, x)
                    if v == 0 and x > removed:
                        selected = temp_selected + [x]
                        selected_set = temp_set | {x}
                        available.remove(x)
                        available.append(removed)
                        improved = True
                        break
                if improved:
                    break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s limit)
    
    best = []
    
    # Strategy 1: Largest-first greedy (fast, good baseline)
    result = _greedy_largest_first(deadline)
    if len(result) > len(best):
        best = result
    
    # Strategy 2: Priority-based greedy
    if time.time() < deadline - 5:
        result = _greedy_with_priority(deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Random restarts with different orderings
    random.seed(12345)
    for trial in range(30):
        if time.time() >= deadline - 5:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        
        selected = []
        selected_set = set()
        for x in order:
            if len(selected) >= 48:
                break
            v = _count_violations_for_new(selected, x)
            if v == 0:
                selected.append(x)
                selected_set.add(x)
        
        if len(selected) > len(best):
            best = selected
    
    # Strategy 4: Local search on best result
    if len(best) < 48 and time.time() < deadline - 2:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if _check_validity(best):
        return best
    
    # Fallback: return largest-first result if valid
    fallback = _greedy_largest_first(deadline)
    if _check_validity(fallback):
        return fallback
    
    return []

# EVOLVE-BLOCK-END

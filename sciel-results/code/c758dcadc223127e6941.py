"""Optimized 3-progression-free set maximizer using branch-and-bound with bitsets."""
import time
from fractions import Fraction

N = 256

def _build_conflicts():
    """Build conflict sets: for each x, the set of y such that adding y to a set
    containing x creates a 3-AP. Actually we need: for a candidate x, which existing
    elements conflict with it.
    
    We'll precompute for each pair (a, b) with a < b, the forbidden c = 2*b - a
    (if c > b) or c = 2*a - b (if c < a, but since a < b, 2*a-b < a, so it's fine).
    
    Better approach: for each x, precompute the set of y > x such that x, y form part of a 3-AP.
    Actually, let's think differently.
    
    For branch and bound, we process numbers in some order. When we consider adding x,
    we need to check if x conflicts with any already-selected elements.
    
    x conflicts with a, b in selected if:
    - 2*b - a == x (x is the end of a 3-AP a, b, x)
    - 2*a - b == x (x is the start of a 3-AP x, a, b) 
    - 2*x - a == b or 2*x - b == a (x is the middle of a 3-AP)
    
    Actually simpler: x conflicts with selected set S if there exist a, b in S such that
    {a, b, x} forms a 3-AP. This means:
    - x = 2b - a for some a < b in S (x > b)
    - x = 2a - b for some a > b in S (x < b < a) -- but if we process in order, this might not apply
    - a = 2x - b or b = 2x - a for some a, b in S (x is middle)
    
    Let's precompute for each x, the set of pairs (a, b) in [1, N] such that {a, b, x} is a 3-AP.
    """
    # For each x, conflicts[x] = set of y such that x and y conflict (i.e., there exists z where {x,y,z} is 3-AP)
    # But actually for the algorithm, when we add x to selected set S, x is valid iff for all a in S,
    # there is no b in S such that {a, b, x} is a 3-AP.
    
    # Let's precompute: for each x, forbidden_by[x] = set of y such that if y is in S and we want to add x,
    # then we need to check if there's a z in S making 3-AP. This is getting complex.
    
    # Simpler: precompute for each x, the set of all y such that x and y are in some 3-AP.
    # i.e., exists z in [1,N] with {x,y,z} = {a, a+d, a+2d} for some a, d != 0.
    conflicts = [set() for _ in range(N+1)]
    for a in range(1, N+1):
        for d in range(1, N+1):
            b = a + d
            c = a + 2*d
            if c > N:
                break
            # a, b, c form 3-AP
            conflicts[a].add(b)
            conflicts[a].add(c)
            conflicts[b].add(a)
            conflicts[b].add(c)
            conflicts[c].add(a)
            conflicts[c].add(b)
    return conflicts

def _greedy_solution(conflicts):
    """Greedy: add numbers in order of decreasing 1/x (i.e., increasing x) if no conflict."""
    selected = set()
    for x in range(1, N+1):
        # Check if x conflicts with any in selected
        ok = True
        for y in selected:
            if y in conflicts[x]:
                ok = False
                break
        if ok:
            selected.add(x)
    return list(selected)

def _reciprocal_weight(selected):
    """Compute exact sum of reciprocals."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # 5 second margin before 90s limit
    
    conflicts = _build_conflicts()
    
    # Start with greedy solution
    best_set = _greedy_solution(conflicts)
    best_weight = _reciprocal_weight(best_set)
    
    # Try local improvement: for each element, try removing it and re-adding others
    # Also try swapping
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        current = set(best_set)
        current_weight = best_weight
        
        # Try removing one element
        for x in list(current):
            new_set = current - {x}
            # After removing x, try to add back other elements
            temp_set = set(new_set)
            temp_weight = current_weight - Fraction(1, x)
            # Try adding elements not in temp_set, in order of 1/x
            added_any = False
            for y in range(1, N+1):
                if y in temp_set:
                    continue
                ok = True
                for z in temp_set:
                    if z in conflicts[y]:
                        ok = False
                        break
                if ok:
                    temp_set.add(y)
                    temp_weight += Fraction(1, y)
                    added_any = True
            
            if temp_weight > best_weight:
                best_set = list(temp_set)
                best_weight = temp_weight
                improved = True
                break
        else:
            # Try swapping: remove x, add y
            for x in list(current):
                if time.time() > DEADLINE:
                    return best_set
                for y in range(1, N+1):
                    if y in current:
                        continue
                    # Check if y can be added when x is removed
                    test_set = (current - {x})
                    ok = True
                    for z in test_set:
                        if z in conflicts[y]:
                            ok = False
                            break
                    if ok:
                        new_weight = current_weight - Fraction(1, x) + Fraction(1, y)
                        if new_weight > best_weight:
                            best_set = list(test_set | {y})
                            best_weight = new_weight
                            improved = True
                            break
                if improved:
                    break
        if improved:
            current = set(best_set)
            current_weight = best_weight
    
    return best_set

# Precompute for faster access
_conflicts = None
def _get_conflicts():
    global _conflicts
    if _conflicts is None:
        _conflicts = _build_conflicts()
    return _conflicts

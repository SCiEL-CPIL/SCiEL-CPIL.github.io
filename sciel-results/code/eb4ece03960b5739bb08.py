"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _valid_addition(selected, x):
    """Check if x can be added to selected without violating the constraint."""
    # For each a in selected + [x], check that a does not divide sum of any 3 distinct OTHER elements
    candidate = selected + [x]
    n = len(candidate)
    # Precompute for speed: for each pair (i,j) with i<j, we need sums with a third
    # Actually, for each element a in candidate, we need to check all triples from candidate\{a}
    # This is O(n^3) per check, but n is at most ~48, so n^3 ~ 110k, times 192 candidates = ~21M ops
    # Let's be more careful.
    
    # For the new element x: check that x doesn't divide sum of any 3 from selected
    # For each existing a in selected: check that a doesn't divide sum of any 3 from (selected \ {a}) ∪ {x}
    
    # Check x: x should not divide sum of any 3 distinct elements from selected
    if len(selected) >= 3:
        for t in combinations(selected, 3):
            if sum(t) % x == 0:
                return False
    
    # Check each existing a in selected: a should not divide sum of any 3 from (selected - {a}) + {x}
    for a in selected:
        others = [v for v in selected if v != a]
        # Triples can be: 3 from others, or 2 from others + x
        # 3 from others: already valid since selected was valid
        # 2 from others + x: need to check
        for pair in combinations(others, 2):
            s = pair[0] + pair[1] + x
            if s % a == 0:
                return False
    return True

def _is_valid_set(selected):
    """Verify the entire set is valid."""
    n = len(selected)
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        if len(others) >= 3:
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
    return True

def _greedy_largest_first():
    """Original greedy: pick largest available first."""
    selected = []
    for x in range(192, 0, -1):
        if _valid_addition(selected, x):
            selected.append(x)
    return selected

def _greedy_degree_order():
    """
    Greedy with degree-based ordering: at each step, pick the candidate that
    creates the fewest future conflicts (lowest 'degree' in conflict hypergraph).
    """
    selected = []
    available = list(range(1, 193))
    
    while available:
        best_x = None
        best_score = -1
        for x in available:
            if not _valid_addition(selected, x):
                continue
            # Score: how many remaining candidates would be blocked by adding x
            # Approximate by counting how many y in available (y != x) would become invalid
            # This is expensive, so use a lighter heuristic:
            # Count conflicts: for each a in selected + [x], count pairs in available that with x or another create divisible sums
            # Simpler: just pick the one with smallest value (greedy small-first) or use a proxy
            # Let's use: prefer elements that are less likely to divide sums -> larger elements are safer as divisors
            # But also, x as a summand: smaller x is less likely to make sums divisible
            # Heuristic: score = -x (prefer smaller) or use actual conflict count
            # For speed, use a simple heuristic: prefer larger x (less likely to divide sums)
            score = x  # prefer larger
            if score > best_score:
                best_score = score
                best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        available.remove(best_x)
    
    return selected

def _local_search(selected, max_iters=5000):
    """
    Local search: try to add one element or swap to improve cardinality.
    """
    selected = list(selected)
    available = [x for x in range(1, 193) if x not in selected]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try to add any available element
        for x in available:
            if _valid_addition(selected, x):
                selected.append(x)
                available.remove(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add two
        # This is O(n^2 * m) which could be slow. Limit to a few attempts.
        # Try: for each pair to remove, try to add one that fits
        n = len(selected)
        if n >= 2:
            # Try removing one element and adding two
            for i in range(n):
                for j in range(i+1, n):
                    new_selected = selected[:i] + selected[i+1:j] + selected[j+1:]
                    # Try to add two from available
                    added = 0
                    for x in available:
                        if _valid_addition(new_selected, x):
                            new_selected.append(x)
                            added += 1
                            if added == 2:
                                break
                    if len(new_selected) > n:
                        selected = new_selected
                        available = [x for x in range(1, 193) if x not in selected]
                        improved = True
                        break
                if improved:
                    break
    return selected

def _randomized_greedy(seed):
    """
    Randomized greedy: shuffle candidates, use degree-based scoring.
    """
    rng = random.Random(seed)
    selected = []
    available = list(range(1, 193))
    rng.shuffle(available)
    
    # Process in shuffled order but with some priority
    for x in available:
        if _valid_addition(selected, x):
            selected.append(x)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Start with the known good greedy
    best = _greedy_largest_first()
    best_size = len(best)
    
    # Try local search on the initial solution
    if time.time() < deadline:
        improved = _local_search(best, max_iters=2000)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Try multiple randomized greedy + local search
    for seed in range(1, 1000):
        if time.time() >= deadline:
            break
        
        sol = _randomized_greedy(seed)
        # Quick local search
        if time.time() < deadline:
            sol = _local_search(sol, max_iters=500)
        
        if len(sol) > best_size:
            best = sol
            best_size = len(sol)
    
    # Final local search on best
    if time.time() < deadline:
        best = _local_search(best, max_iters=3000)
    
    return best

# EVOLVE-BLOCK-END

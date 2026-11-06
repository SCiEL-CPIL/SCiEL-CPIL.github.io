# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _is_valid(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    if n < 4:
        return True
    sel = np.array(selected, dtype=np.int64)
    for i in range(n):
        a = sel[i]
        others = np.delete(sel, i)
        # Check all triples in others
        for j in range(len(others)):
            for k in range(j+1, len(others)):
                for l in range(k+1, len(others)):
                    if (others[j] + others[k] + others[l]) % a == 0:
                        return False
    return True

def _can_add_fast(selected, x):
    """
    Fast check if x can be added to selected.
    1. No triple in selected sums to multiple of x.
    2. For each a in selected, no pair (s1,s2) in selected\{a} with x such that (s1+s2+x)%a==0.
    """
    n = len(selected)
    
    # Check 1: Triples in selected sum to multiple of x
    if n >= 3:
        sel = np.array(selected, dtype=np.int64)
        # Vectorized: check all C(n,3) triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (sel[i] + sel[j] + sel[k]) % x == 0:
                        return False
    
    # Check 2: For each a in selected, check if x creates a bad triple with two others
    for i in range(n):
        a = selected[i]
        # Need distinct s1, s2 from selected \ {a} such that (s1 + s2 + x) % a == 0
        # i.e., (s1 + s2) % a == (-x) % a
        target = (-x) % a
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        if m < 2:
            continue
        # Check all pairs in others
        for j in range(m):
            for k in range(j+1, m):
                if (others[j] + others[k]) % a == target:
                    return False
    
    return True

def _greedy(order, deadline):
    """Greedy construction with given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(selected, x):
            selected.append(x)
    return selected

def _local_improve(selected, deadline):
    """Local search: try to add more elements or swap."""
    selected = list(selected)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding elements
        candidates = [x for x in range(1, 193) if x not in set(selected)]
        candidates.sort(reverse=True)
        for x in candidates:
            if time.time() > deadline:
                break
            if _can_add_fast(selected, x):
                selected.append(x)
                improved = True
                break  # Restart to re-evaluate
        
        # 2. Try swaps (remove 1, add 1 or 2)
        if not improved and time.time() < deadline:
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = selected[:i] + selected[i+1:]
                
                # Try to add 2 elements to remaining
                added = []
                candidates = [x for x in range(1, 193) if x not in set(remaining) and x != removed]
                candidates.sort(reverse=True)
                for x in candidates:
                    if time.time() > deadline:
                        break
                    if len(added) == 2:
                        break
                    test_set = remaining + added
                    if _can_add_fast(test_set, x):
                        added.append(x)
                
                if len(added) >= 1:
                    new_selected = remaining + added
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        improved = True
                        break

    return selected

def _graph_mis_heuristic(deadline):
    """
    Build a conflict graph where vertices are 1..192.
    Two numbers conflict if they cannot both be in a valid set.
    Use greedy coloring / independent set heuristic.
    """
    n_max = 192
    # Precompute conflicts: for each pair (a,b), check if they conflict
    # a and b conflict if there exists a third number c such that:
    #   (a+b+c) % a == 0 or (a+b+c) % b == 0
    # But this depends on c being in the set, so it's not a simple pairwise conflict.
    # Instead, use a different approach: greedy with prioritization.
    
    # Heuristic: prioritize numbers that are "harder to place" (more constraints)
    # Numbers with many divisors create more conflicts.
    
    # Score each number by how many triples it would block
    scores = {}
    for x in range(1, n_max + 1):
        # Count how many pairs (a,b) with a,b < x or a,b > x would have (a+b+x)%x==0
        # or more practically, just use a heuristic
        # Numbers with many divisors are more constrained
        div_count = sum(1 for d in range(1, x+1) if x % d == 0)
        scores[x] = div_count
    
    # Sort by score descending (most constrained first)
    order = sorted(range(1, n_max + 1), key=lambda x: -scores[x])
    
    result = _greedy(order, deadline)
    return result

def _random_greedy(deadline):
    """Randomized greedy with multiple restarts."""
    best = []
    start_time = time.time()
    
    # Try several random orderings
    max_attempts = 100
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        rand_order = list(range(1, 193))
        random.shuffle(rand_order)
        result = _greedy(rand_order, deadline)
        if len(result) > len(best):
            best = result
    
    # Also try structured orderings
    structured_orders = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
        # Graph MIS heuristic order
    ]
    
    # Add the graph MIS heuristic
    graph_result = _graph_mis_heuristic(deadline)
    if len(graph_result) > len(best):
        best = graph_result
    
    for order in structured_orders:
        if time.time() > deadline:
            break
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin for 90s limit
    
    # Phase 1: Randomized greedy to find good initial solution
    best = _random_greedy(deadline)
    
    # Phase 2: Local improvement
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 3: More local improvement rounds
    if time.time() < deadline - 5:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
            
    # Phase 4: More random restarts with local improvement
    if time.time() < deadline - 2:
        current_best = best
        current_len = len(best)
        for _ in range(30):
            if time.time() > deadline - 2:
                break
            rand_order = list(range(1, 193))
            random.shuffle(rand_order)
            result = _greedy(rand_order, deadline)
            if len(result) > current_len:
                # Local improve the new best
                improved = _local_improve(result, deadline)
                if len(improved) > current_len:
                    current_best = improved
                    current_len = len(improved)
        best = current_best

    # Final validation
    if not _is_valid(best):
        # Fallback to simple greedy
        best = _greedy(list(range(192, 0, -1)), deadline)
    
    return best
# EVOLVE-BLOCK-END

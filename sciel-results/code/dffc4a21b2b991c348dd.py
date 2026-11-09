"""Rewritten program: precompute bad-triple conflicts, run many greedy+local-search restarts."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_conflicts(n=128):
    """
    Precompute for each x, the set of pairs (a,b) such that {a,b,x} is a bad triple.
    A bad triple has all three pairwise LCMs equal.
    """
    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    forbidden_pairs = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for a in range(1, n + 1):
            if a == x:
                continue
            for b in range(a + 1, n + 1):
                if b == x:
                    continue
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    forbidden_pairs[x].add((a, b))
    
    return forbidden_pairs

def _can_add(x, selected, forbidden_pairs):
    """Check if x can be added to selected set."""
    for (a, b) in forbidden_pairs[x]:
        if a in selected and b in selected:
            return False
    return True

def _greedy_select(forbidden_pairs, order):
    """Greedily select elements in the given order."""
    selected = set()
    for x in order:
        if _can_add(x, selected, forbidden_pairs):
            selected.add(x)
    return selected

def _local_search(forbidden_pairs, selected_set, max_iters=50):
    """
    1-exchange local search: try removing one element and adding another.
    """
    best_set = set(selected_set)
    best_size = len(best_set)
    
    for _ in range(max_iters):
        improved = False
        all_elems = list(best_set)
        for elem_to_remove in all_elems:
            trial_set = best_set - {elem_to_remove}
            for x in range(1, 129):
                if x in trial_set:
                    continue
                if _can_add(x, trial_set, forbidden_pairs):
                    new_set = trial_set | {x}
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_size = len(new_set)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best_set

def construct():
    deadline = time.time() + 78  # 78s budget, leaving 12s margin for safety
    
    forbidden_pairs = _precompute_conflicts(128)
    
    best_set = set()
    best_size = 0
    
    # Baseline: ascending greedy
    order = list(range(1, 129))
    baseline = _greedy_select(forbidden_pairs, order)
    best_set = baseline
    best_size = len(baseline)
    
    # Also try descending
    order_desc = list(range(128, 0, -1))
    desc_set = _greedy_select(forbidden_pairs, order_desc)
    if len(desc_set) > best_size:
        best_set = desc_set
        best_size = len(desc_set)
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Randomized order
        order = list(range(1, 129))
        rng.shuffle(order)
        
        candidate = _greedy_select(forbidden_pairs, order)
        
        if len(candidate) > best_size:
            best_set = candidate
            best_size = len(candidate)
        
        # Local search on promising candidates
        if len(candidate) >= best_size - 3:
            improved = _local_search(forbidden_pairs, candidate, max_iters=30)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END

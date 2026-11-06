"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

_START_TIME = time.time()
TIME_LIMIT = 82.0

def _can_represent(target, candidates):
    """Check if target can be represented as sum of distinct unit fractions from candidates."""
    if not candidates:
        return target == 0
    candidates = sorted(candidates, reverse=True)
    n = len(candidates)
    
    # Precompute suffix sums for pruning
    suffix = [Fraction(0, 1)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + Fraction(1, candidates[i])
    
    def dfs(idx, remaining):
        if remaining == 0:
            return True
        if idx >= n:
            return False
        if remaining < 0:
            return False
        if suffix[idx] < remaining:
            return False
        
        # Try taking candidates[idx]
        frac = Fraction(1, candidates[idx])
        if frac <= remaining:
            if dfs(idx + 1, remaining - frac):
                return True
        
        # Try skipping
        return dfs(idx + 1, remaining)
    
    return dfs(0, target)

def _is_valid(selected):
    """Check if selected set has no reciprocal-subsum conflicts."""
    selected_set = set(selected)
    for a in selected_set:
        target = Fraction(1, a)
        candidates = [s for s in selected_set if s > a]
        if _can_represent(target, candidates):
            return False
    return True

def _can_add(a, selected_set):
    """Check if adding a to selected_set creates any conflict."""
    target_a = Fraction(1, a)
    candidates_a = [s for s in selected_set if s > a]
    if _can_represent(target_a, candidates_a):
        return False
    
    for b in selected_set:
        target_b = Fraction(1, b)
        candidates_b = [s for s in selected_set if s > b]
        if a > b:
            candidates_b.append(a)
        if _can_represent(target_b, candidates_b):
            return False
    return True

def _greedy_build(order):
    selected = set()
    for a in order:
        if _can_add(a, selected):
            selected.add(a)
    return selected

def _local_search(selected, all_elements, max_iters=50000):
    selected = set(selected)
    best = len(selected)
    
    for _ in range(max_iters):
        if time.time() - _START_TIME > TIME_LIMIT:
            break
        improved = False
        
        for a in all_elements:
            if a not in selected and _can_add(a, selected):
                selected.add(a)
                improved = True
                if len(selected) > best:
                    best = len(selected)
                break
        
        if not improved:
            for remove in list(selected):
                found = False
                for a1 in all_elements:
                    if a1 in selected:
                        continue
                    for a2 in all_elements:
                        if a2 in selected or a2 <= a1:
                            continue
                        new_set = (selected - {remove}) | {a1, a2}
                        if _is_valid(new_set):
                            selected = new_set
                            improved = True
                            if len(selected) > best:
                                best = len(selected)
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return selected

def construct():
    random.seed(99999)
    best_set = set()
    all_elements = list(range(1, 25))
    
    orders = [
        list(range(24, 0, -1)),
        list(range(1, 25)),
    ]
    
    for _ in range(50):
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() - _START_TIME > TIME_LIMIT:
            break
        s = _greedy_build(order)
        s = _local_search(s, all_elements)
        if len(s) > len(best_set):
            best_set = set(s)
    
    if best_set:
        best_set = _local_search(best_set, all_elements, max_iters=100000)
    
    if not best_set:
        best_set = {24}
    
    return sorted(best_set)

# EVOLVE-BLOCK-END

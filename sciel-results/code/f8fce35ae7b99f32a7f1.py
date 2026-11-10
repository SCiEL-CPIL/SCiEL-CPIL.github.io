import time
from fractions import Fraction

def _find_solution_n(n_terms, min_den=5, max_den=10000000, deadline=None):
    if deadline is None:
        deadline = time.time() + 50
    
    target = Fraction(1, 1)
    dens = []
    
    def search(idx, current_sum, last_den):
        if time.time() > deadline:
            return None
        
        if idx == n_terms:
            if current_sum == target:
                return dens.copy()
            return None
        
        remaining = n_terms - idx
        
        # Pruning
        max_possible = sum(Fraction(1, last_den + 1 + i) for i in range(remaining))
        if current_sum + max_possible < target:
            return None
            
        min_possible = sum(Fraction(1, max_den - i) for i in range(remaining))
        if current_sum + min_possible > target:
            return None
            
        R = target - current_sum
        if R <= 0:
            return None
            
        low_d = max(last_den + 1, int(Fraction(1, 1) / R) + 1)
        
        min_remaining_sum = Fraction(remaining - 1, max_den)
        if R <= min_remaining_sum:
            return None
        high_d = min(max_den, int(Fraction(1, 1) / (R - min_remaining_sum)))
        
        for d in range(low_d, high_d + 1):
            if time.time() > deadline:
                return None
            dens.append(d)
            result = search(idx + 1, current_sum + Fraction(1, d), d)
            if result is not None:
                return result
            dens.pop()
            
        return None

    return search(0, Fraction(0, 1), min_den - 1)

def construct():
    deadline = time.time() + 55
    
    res = _find_solution_n(10, deadline=deadline)
    if res is not None:
        return res
        
    res = _find_solution_n(11, deadline=deadline)
    if res is not None:
        return res
        
    # Fallback DP
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no solution found")

import time
import random
from fractions import Fraction

def _check_addition(sums, new_frac):
    """Check if adding new_frac keeps all subset sums distinct.
    sums: list of current subset sums (as Fractions)
    new_frac: the new reciprocal value
    Returns True if no collision.
    """
    sums_set = set(sums)
    for s in sums:
        if s + new_frac in sums_set:
            return False
    return True

def _compute_sums(fracs):
    """Compute all subset sums for a list of fractions."""
    sums = [Fraction(0)]
    for r in fracs:
        sums += [s + r for s in sums]
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # Seed fallback
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Verify seed is valid
    seed_fracs = [Fraction(1, x) for x in best_set]
    seed_sums = _compute_sums(seed_fracs)
    if len(set(seed_sums)) != len(seed_sums):
        best_set = [1]
        best_size = 1
    
    random.seed(42)
    
    # Greedy with many random orderings
    for trial in range(2000):
        if time.time() > deadline - 5:
            break
        
        order = list(range(1, 23))
        random.shuffle(order)
        
        current_fracs = []
        current_sums = [Fraction(0)]
        current_elements = []
        
        for x in order:
            if time.time() > deadline - 5:
                break
            new_frac = Fraction(1, x)
            if _check_addition(current_sums, new_frac):
                current_fracs.append(new_frac)
                current_sums += [s + new_frac for s in current_sums]
                current_elements.append(x)
        
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best_set = list(current_elements)
    
    # Try to extend the best set by adding remaining elements
    if time.time() < deadline - 10:
        best_fracs = [Fraction(1, x) for x in best_set]
        best_sums = _compute_sums(best_fracs)
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for x in range(1, 23):
                if x in best_set:
                    continue
                new_frac = Fraction(1, x)
                if _check_addition(best_sums, new_frac):
                    best_fracs.append(new_frac)
                    best_sums += [s + new_frac for s in best_sums]
                    best_set.append(x)
                    changed = True
                    break
    
    return best_set

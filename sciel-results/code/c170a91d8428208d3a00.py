import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recips_frac = [Fraction(1, d) for d in denominators]
    
    best_deficit = None
    best_indices = None
    
    def try_combo(combo):
        nonlocal best_deficit, best_indices
        s = sum(recips_frac[i] for i in combo)
        if s < 1:
            deficit = 1 - s
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_indices = list(combo)
    
    # Exhaustive search for sizes 3-5
    for size in range(3, 6):
        if time.time() > deadline:
            break
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            s = sum(recips_frac[i] for i in combo)
            if s < 1:
                deficit = 1 - s
                if best_deficit is None or deficit < best_deficit:
                    best_deficit = deficit
                    best_indices = list(combo)
    
    # For size 6, do a more targeted search
    if time.time() < deadline:
        for combo in combinations(range(n), 6):
            if time.time() > deadline:
                break
            s = sum(recips_frac[i] for i in combo)
            if s < 1:
                deficit = 1 - s
                if best_deficit is None or deficit < best_deficit:
                    best_deficit = deficit
                    best_indices = list(combo)
    
    if best_indices is None:
        return [0]
    
    return best_indices

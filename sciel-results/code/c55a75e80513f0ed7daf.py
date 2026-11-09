"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _is_dissociated(indices):
    """Check if all subset sums of reciprocals are distinct using exact fractions."""
    n = len(indices)
    if n == 0:
        return True
    if n > 20:
        # For very large sets, we can't check all 2^n sums
        # Use a probabilistic or heuristic check
        # Actually for n up to 22, 2^22 = 4M which is feasible
        pass
    
    seen = set()
    for mask in range(1 << n):
        s = Fraction(0, 1)
        for i in range(n):
            if mask & (1 << i):
                s += Fraction(1, indices[i])
        if s in seen:
            return False
        seen.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # The known good set from seed: [1, 2, 3, 5, 7, 11, 13, 17, 19] size 9
    # Try to find larger sets
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best)
    
    # Try to add elements to the current best
    # and use local search
    
    current = list(best)
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        # Try adding a random element
        available = [x for x in range(1, 23) if x not in current]
        if available:
            add_elem = random.choice(available)
            new_set = current + [add_elem]
            if _is_dissociated(new_set):
                current = new_set
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
        
        # Try swapping
        if len(current) > 1:
            ri = random.randint(0, len(current) - 1)
            remove_elem = current[ri]
            available = [x for x in range(1, 23) if x not in current]
            if available:
                add_elem = random.choice(available)
                new_set = [x for i, x in enumerate(current) if i != ri]
                new_set.append(add_elem)
                if _is_dissociated(new_set):
                    current = new_set
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
        
        # Try remove one, add two
        if len(current) > 2:
            ri = random.randint(0, len(current) - 1)
            remove_elem = current[ri]
            new_set = [x for i, x in enumerate(current) if i != ri]
            available = [x for x in range(1, 23) if x not in new_set]
            random.shuffle(available)
            added_count = 0
            for ae in available:
                if added_count >= 2:
                    break
                test = new_set + [ae]
                if _is_dissociated(test):
                    new_set = test
                    added_count += 1
            if len(new_set) > len(current):
                current = new_set
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
    
    # Final check
    if not _is_dissociated(best):
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return sorted(best)
# EVOLVE-BLOCK-END

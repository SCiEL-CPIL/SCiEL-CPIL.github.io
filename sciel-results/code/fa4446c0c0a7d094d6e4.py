import time
import random
from itertools import combinations

def _is_sidon_4(subset):
    """Check if a 4-element subset is Sidon."""
    a, b, c, d = subset
    sums = {
        2*a, a+b, a+c, a+d,
        2*b, b+c, b+d,
        2*c, c+d,
        2*d
    }
    return len(sums) == 10

def _has_sidon_4(arr):
    """Check if arr contains any 4-element Sidon subset."""
    for combo in combinations(arr, 4):
        if _is_sidon_4(combo):
            return True
    return False

def _count_sidon_4(arr):
    """Count number of 4-element Sidon subsets."""
    count = 0
    for combo in combinations(arr, 4):
        if _is_sidon_4(combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Phase 1: Brute force small ranges
    # We look for a set with 0 Sidon-4 subsets
    for max_val in [15, 18, 20, 22, 25]:
        if time.time() > deadline - 10: # Keep 10s for local search
            break
        for cand in combinations(range(max_val + 1), 12):
            if not _has_sidon_4(cand):
                return list(cand)
            
    # Phase 2: Local Search
    # Minimize count of Sidon-4 subsets
    best_set = list(range(12)) # Default
    best_count = _count_sidon_4(best_set)
    
    random.seed(42)
    current_set = best_set[:]
    current_count = best_count
    
    # Start with a random set in a dense range
    current_set = random.sample(range(0, 40), 12)
    current_set.sort()
    current_count = _count_sidon_4(current_set)
    if current_count < best_count:
        best_set = current_set[:]
        best_count = current_count
    
    T = 1.0 # Temperature
    while time.time() < deadline - 2:
        # Mutation
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        # Try a new value
        # Bias towards values close to existing ones or small values
        new_val = random.randint(0, 50)
        if new_val == old_val:
            continue
        if new_val in current_set:
            # Find a non-colliding value or skip
            # Simple retry
            attempts = 0
            while new_val in current_set and attempts < 10:
                new_val = random.randint(0, 50)
                attempts += 1
            if new_val in current_set:
                continue
                
        # Apply mutation
        current_set[idx] = new_val
        current_set.sort()
        
        new_count = _count_sidon_4(current_set)
        
        if new_count < current_count:
            current_count = new_count
            if new_count < best_count:
                best_count = new_count
                best_set = current_set[:]
                if best_count == 0:
                    return best_set
        else:
            # Metropolis criterion
            delta = new_count - current_count
            if delta == 0:
                # Accept to allow exploration
                current_count = new_count
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < pow(2.71828, -delta / T):
                    current_count = new_count
                else:
                    # Revert
                    current_set[idx] = old_val
                    current_set.sort()
        
        # Cool down
        T *= 0.999
        
    return best_set

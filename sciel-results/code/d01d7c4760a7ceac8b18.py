"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _count_collisions(arr):
    """Count number of unordered pair sum collisions (including diagonals)."""
    sums = {}
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sums[s] = sums.get(s, 0) + 1
    return sum(c * (c - 1) // 2 for c in sums.values())

def _max_sidon_size(arr):
    """Find the largest subset with all pair sums unique (including diagonals)."""
    n = len(arr)
    # Try from largest subset size down
    for size in range(n, 1, -1):
        for subset in combinations(range(n), size):
            # Check if this subset is Sidon (all unordered pair sums unique incl. diagonals)
            sums = set()
            valid = True
            for i in range(size):
                for j in range(i, size):
                    s = arr[subset[i]] + arr[subset[j]]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return size
    return 1

def _max_sidon_size_fast(arr):
    """Optimized: try subsets greedily, then verify."""
    n = len(arr)
    # First check if the full set is Sidon
    sums = set()
    valid = True
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    if valid:
        return n
    
    # Try size n-1
    for skip in range(n):
        subset = [arr[i] for i in range(n) if i != skip]
        sums = set()
        valid = True
        for i in range(n - 1):
            for j in range(i, n - 1):
                s = subset[i] + subset[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        if valid:
            return n - 1
    
    # Try size n-2
    for skip1 in range(n):
        for skip2 in range(skip1 + 1, n):
            subset = [arr[i] for i in range(n) if i != skip1 and i != skip2]
            sums = set()
            valid = True
            for i in range(n - 2):
                for j in range(i, n - 2):
                    s = subset[i] + subset[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
            if valid:
                return n - 2
    
    # Fallback: exhaustive from size 5 down
    for size in range(n - 3, 1, -1):
        for subset in combinations(range(n), size):
            sums = set()
            valid = True
            for i in range(size):
                for j in range(i, size):
                    s = arr[subset[i]] + arr[subset[j]]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Start with a structured initial set that has many collisions
    # Use arithmetic progression-like structure to create sum collisions
    best_arr = list(range(0, 128, 128 // 12))[:12]
    # Ensure distinct
    seen = set()
    unique_best = []
    for x in best_arr:
        if x not in seen:
            seen.add(x)
            unique_best.append(x)
    while len(unique_best) < 12:
        unique_best.append(random.randint(0, 128))
        seen.add(unique_best[-1])
    best_arr = unique_best[:12]
    
    best_score = _max_sidon_size_fast(best_arr)
    
    # Local search: swap-based
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Try random swaps
        i, j = random.sample(range(12), 2)
        old_i, old_j = best_arr[i], best_arr[j]
        
        # Try swapping with a random value
        new_val = random.randint(0, 128)
        
        # Try replacing element i
        candidate = best_arr[:]
        candidate[i] = new_val
        if len(set(candidate)) == 12:
            score = _max_sidon_size_fast(candidate)
            if score < best_score:
                best_arr = candidate
                best_score = score
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Try swapping elements i and j with new values
        candidate = best_arr[:]
        candidate[i] = new_val
        candidate[j] = random.randint(0, 128)
        if len(set(candidate)) == 12:
            score = _max_sidon_size_fast(candidate)
            if score < best_score:
                best_arr = candidate
                best_score = score
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Try pure swap of two elements
        candidate = best_arr[:]
        candidate[i], candidate[j] = candidate[j], candidate[i]
        score = _max_sidon_size_fast(candidate)
        if score < best_score:
            best_arr = candidate
            best_score = score
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_arr
# EVOLVE-BLOCK-END

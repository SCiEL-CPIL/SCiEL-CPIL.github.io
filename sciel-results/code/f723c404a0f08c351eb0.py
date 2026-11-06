"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import itertools
from collections import Counter

# EVOLVE-BLOCK-START
def _collision_score(aset):
    """Count the number of sum collisions in the set (higher = more collisions = better for us)."""
    n = len(aset)
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(aset[i] + aset[j])
    c = Counter(sums)
    # Number of collisions: for each sum that appears k times, it contributes k*(k-1)/2 collisions
    collisions = 0
    for cnt in c.values():
        collisions += cnt * (cnt - 1) // 2
    return collisions

def _is_sidon_subset(subset):
    """Check if a subset has all pair sums (including diagonals) unique."""
    n = len(subset)
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(subset[i] + subset[j])
    return len(sums) == len(set(sums))

def _max_sidon_size(aset):
    """Find the maximum size of a Sidon subset of aset."""
    n = len(aset)
    # Try subsets from largest to smallest
    for size in range(n, 0, -1):
        for combo in itertools.combinations(range(n), size):
            subset = [aset[i] for i in combo]
            if _is_sidon_subset(subset):
                return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Initial random set
    best_set = list(range(12))
    best_exact = _max_sidon_size(best_set)
    best_collision = _collision_score(best_set)
    
    current_set = best_set[:]
    current_collision = best_collision
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Perturb: pick a random element and change it
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        new_val = random.randint(0, 128)
        
        # Try a few random values to find one that maintains distinctness
        found = False
        for _ in range(20):
            new_val = random.randint(0, 128)
            if new_val != old_val and new_val not in current_set:
                found = True
                break
        
        if not found:
            continue
        
        current_set[idx] = new_val
        new_collision = _collision_score(current_set)
        
        delta = new_collision - current_collision
        
        # Accept if better or with probability exp(delta/T) for worse
        if delta > 0 or (T > T_min and random.random() < pow(2.718281828, delta / T)):
            current_collision = new_collision
            # Periodically check exact value
            if steps % 200 == 0:
                exact = _max_sidon_size(current_set)
                if exact < best_exact:
                    best_exact = exact
                    best_set = current_set[:]
                    best_collision = current_collision
        else:
            current_set[idx] = old_val
        
        T *= cooling
        
        # If we found a very good set, do local search around it
        if best_exact <= 5:
            # Try to improve further with more iterations
            pass
    
    # Final local search on best_set
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(12):
                for new_val in range(0, 129):
                    if new_val == best_set[i] or new_val in best_set:
                        continue
                    old = best_set[i]
                    best_set[i] = new_val
                    new_col = _collision_score(best_set)
                    if new_col > best_collision:
                        # Check if it actually improves exact
                        exact = _max_sidon_size(best_set)
                        if exact < best_exact:
                            best_exact = exact
                            best_collision = new_col
                            improved = True
                        else:
                            best_set[i] = old
                    else:
                        best_set[i] = old
    
    return best_set
# EVOLVE-BLOCK-END

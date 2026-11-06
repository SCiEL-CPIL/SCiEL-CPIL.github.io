"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def _compute_collision_graph(elements):
    """Compute collision graph: edge between i,j if they share a sum with another pair."""
    n = len(elements)
    # Map each sum to the set of (i,j) pairs that produce it
    sum_to_pairs = defaultdict(list)
    for i in range(n):
        for j in range(n):
            s = elements[i] + elements[j]
            sum_to_pairs[s].append((i, j))
    
    # An element is "colliding" if any sum it participates in is shared
    # For Sidon: we need all unordered pair sums (including diagonals) unique
    # So for each sum, if multiple pairs produce it, those elements are in conflict
    collision_count = 0
    for s, pairs in sum_to_pairs.items():
        if len(pairs) > 1:
            # Count how many distinct elements are involved in this collision
            involved = set()
            for i, j in pairs:
                involved.add(i)
                involved.add(j)
            collision_count += len(involved)
    return collision_count

def _max_sidon_size(elements):
    """Find the largest subset with all unordered pair sums unique (including diagonals)."""
    n = len(elements)
    best = 0
    # Try all subsets of size from n down to 1, return first valid
    for size in range(n, 0, -1):
        for subset in itertools.combinations(range(n), size):
            # Check if this subset is Sidon
            sums = set()
            valid = True
            for i in range(size):
                for j in range(i, size):
                    s = elements[subset[i]] + elements[subset[j]]
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
    deadline = start_time + 85.0  # Safety margin before 90s
    
    best_elements = list(range(12))
    best_sidon_size = _max_sidon_size(best_elements)
    
    # Simulated annealing to maximize collisions
    current = best_elements[:]
    current_collision = _compute_collision_graph(current)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    random.seed(42)
    
    while time.time() < deadline:
        # Perturb: change one random element
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val in current and new_val != old_val:
            # Try to find a valid replacement
            for _ in range(10):
                new_val = random.randint(0, 128)
                if new_val not in current or new_val == old_val:
                    break
            else:
                continue
        
        current[idx] = new_val
        
        # Check distinctness
        if len(set(current)) != 12:
            current[idx] = old_val
            continue
        
        new_collision = _compute_collision_graph(current)
        delta = new_collision - current_collision
        
        if delta >= 0 or random.random() < min(1.0, (T / T_min) ** 0.01):
            current_collision = new_collision
            # Check if this improves our best Sidon size
            sidon = _max_sidon_size(current)
            if sidon < best_sidon_size:
                best_sidon_size = sidon
                best_elements = current[:]
                if best_sidon_size <= 2:
                    break
        else:
            current[idx] = old_val
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final verification
    if len(set(best_elements)) != 12 or any(x < 0 or x > 128 for x in best_elements):
        return list(range(12))
    
    return best_elements
# EVOLVE-BLOCK-END


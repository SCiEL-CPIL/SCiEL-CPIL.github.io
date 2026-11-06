"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def _max_sidon_size(arr):
    """Find the maximum size of a subset of arr where all unordered pair sums (i<=j) are unique."""
    n = len(arr)
    best = 0
    # Use backtracking with pruning
    # For 12 elements, we can try all subsets in order of size descending
    from itertools import combinations
    
    # Try sizes from n down to 1
    for size in range(n, 0, -1):
        found = False
        for combo in combinations(range(n), size):
            subset = [arr[i] for i in combo]
            # Check if all pair sums (i<=j) are unique
            sums = set()
            valid = True
            for i in range(size):
                for j in range(i, size):
                    s = subset[i] + subset[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return size
    return 0


def _count_collisions(arr):
    """Count the number of sum collisions: pairs (i,j) with i<=j such that a[i]+a[j] = a[k]+a[l] for some other pair."""
    n = len(arr)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s not in sums:
                sums[s] = 0
            sums[s] += 1
    # Number of "extra" sums beyond the first occurrence
    collisions = 0
    for count in sums.values():
        if count > 1:
            collisions += count - 1
    return collisions


def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    rng = random.Random(42)
    
    # Initial state: random 12 distinct integers in [0,128]
    state = list(rng.sample(range(129), 12))
    state.sort()
    
    best_state = list(state)
    best_collisions = _count_collisions(state)
    best_sidon = _max_sidon_size(state)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    current_collisions = best_collisions
    current_sidon = best_sidon
    
    while time.time() < deadline and T > T_min:
        # Perturbation: pick one element and move it
        idx = rng.randint(0, 11)
        old_val = state[idx]
        new_val = rng.randint(0, 128)
        
        if new_val in state:
            T *= cooling
            continue
        
        state[idx] = new_val
        state.sort()
        
        new_collisions = _count_collisions(state)
        delta = new_collisions - current_collisions
        
        if delta >= 0 or rng.random() < math.exp(delta / T):
            current_collisions = new_collisions
            # Update best if collisions improved or sidon size decreased
            if new_collisions > best_collisions:
                best_collisions = new_collisions
                best_state = list(state)
                best_sidon = _max_sidon_size(state)
            elif new_collisions == best_collisions:
                # Check if sidon size is smaller
                if time.time() < deadline:
                    s = _max_sidon_size(state)
                    if s < best_sidon:
                        best_sidon = s
                        best_state = list(state)
                        current_sidon = s
        else:
            # Revert
            state[idx] = old_val
            state.sort()
        
        T *= cooling
    
    # Final local search on best_state to try to reduce Sidon size further
    # Try small perturbations around the best state
    for _ in range(5000):
        if time.time() > deadline:
            break
        cand = list(best_state)
        idx = rng.randint(0, 11)
        old_val = cand[idx]
        new_val = rng.randint(0, 128)
        if new_val in cand:
            continue
        cand[idx] = new_val
        cand.sort()
        c = _count_collisions(cand)
        if c >= best_collisions:
            s = _max_sidon_size(cand)
            if s < best_sidon:
                best_sidon = s
                best_state = cand
                best_collisions = c
    
    return best_state
# EVOLVE-BLOCK-END

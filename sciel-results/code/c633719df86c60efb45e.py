import time
import random
import numpy as np

BASE = 51975

# Precompute all divisors of 51975 greater than 1
def _get_divisors(n):
    divs = set()
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)

DIVISORS = _get_divisors(BASE)

# Precompute index arrays for each (m, r) pair
# For each divisor m and residue r, store the array of indices r, r+m, r+2m, ... < BASE
# This avoids repeated np.arange calls
_INDEX_CACHE = {}
for m in DIVISORS:
    for r in range(m):
        _INDEX_CACHE[(m, r)] = np.arange(r, BASE, m)

def _init_random(deadline):
    """Random initialization."""
    counts = np.zeros(BASE, dtype=np.int8)
    uncovered = BASE
    assignments = {}
    for m in DIVISORS:
        r = random.randint(0, m - 1)
        assignments[m] = r
        idx = _INDEX_CACHE[(m, r)]
        old = counts[idx]
        uncovered -= int(np.sum(old == 0))
        counts[idx] = old + 1
    return counts, uncovered, assignments

def _init_greedy(order, deadline):
    """Greedy initialization: add classes in given order, picking best residue for each."""
    counts = np.zeros(BASE, dtype=np.int8)
    uncovered = BASE
    assignments = {}
    
    for m in order:
        best_r = 0
        best_gain = -1
        # Try each residue for this modulus
        for r in range(m):
            idx = _INDEX_CACHE[(m, r)]
            gain = int(np.sum(counts[idx] == 0))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        assignments[m] = best_r
        idx = _INDEX_CACHE[(m, best_r)]
        old = counts[idx]
        uncovered -= best_gain
        counts[idx] = old + 1
        
        if time.time() > deadline - 10:
            break
    
    return counts, uncovered, assignments

def _local_search(counts, uncovered, assignments, deadline, max_stuck=500):
    """Hill-climbing local search."""
    num_divs = len(DIVISORS)
    stuck = 0
    
    while time.time() < deadline - 5:
        if stuck > max_stuck:
            break
        
        m = DIVISORS[random.randint(0, num_divs - 1)]
        old_r = assignments[m]
        new_r = random.randint(0, m - 1)
        if new_r == old_r:
            continue
        
        old_idx = _INDEX_CACHE[(m, old_r)]
        new_idx = _INDEX_CACHE[(m, new_r)]
        
        # Delta calculation
        old_counts = counts[old_idx]
        new_counts = counts[new_idx]
        
        # Uncovered increases where old_counts == 1 (becomes 0)
        delta = int(np.sum(old_counts == 1))
        # Uncovered decreases where new_counts == 0 (becomes 1)
        delta -= int(np.sum(new_counts == 0))
        
        if delta < 0:
            counts[old_idx] = old_counts - 1
            counts[new_idx] = new_counts + 1
            uncovered += delta
            assignments[m] = new_r
            stuck = 0
        else:
            stuck += 1
    
    return counts, uncovered, assignments

def _simulated_anneal(counts, uncovered, assignments, start_time, deadline):
    """Simulated annealing refinement."""
    num_divs = len(DIVISORS)
    t0 = time.time()
    elapsed = t0 - start_time
    total_time = 150.0
    remaining = total_time - elapsed
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    steps = max(1, int(remaining * 50))  # ~50 steps per second
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, steps - 1))
    
    best_uncovered = uncovered
    best_assignments = assignments.copy()
    
    for step in range(steps):
        if time.time() > deadline - 3:
            break
        
        m = DIVISORS[random.randint(0, num_divs - 1)]
        old_r = assignments[m]
        new_r = random.randint(0, m - 1)
        if new_r == old_r:
            continue
        
        old_idx = _INDEX_CACHE[(m, old_r)]
        new_idx = _INDEX_CACHE[(m, new_r)]
        
        old_counts = counts[old_idx]
        new_counts = counts[new_idx]
        
        delta = int(np.sum(old_counts == 1)) - int(np.sum(new_counts == 0))
        
        if delta < 0:
            # Accept
            counts[old_idx] = old_counts - 1
            counts[new_idx] = new_counts + 1
            uncovered += delta
            assignments[m] = new_r
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_assignments = assignments.copy()
        else:
            # Metropolis criterion
            if T > 0.01:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    counts[old_idx] = old_counts - 1
                    counts[new_idx] = new_counts + 1
                    uncovered += delta
                    assignments[m] = new_r
        
        T *= cooling
    
    # Return best found
    # Reconstruct counts from best_assignments
    best_counts = np.zeros(BASE, dtype=np.int8)
    best_unc = BASE
    for m, r in best_assignments.items():
        idx = _INDEX_CACHE[(m, r)]
        old = best_counts[idx]
        best_unc -= int(np.sum(old == 0))
        best_counts[idx] = old + 1
    
    return best_counts, best_unc, best_assignments

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    random.seed(42)
    
    best_uncovered = BASE
    best_assignments = {}
    
    # Generate multiple initializations
    inits = []
    
    # 1. Ascending greedy
    counts, unc, assign = _init_greedy(DIVISORS, deadline)
    inits.append((counts, unc, assign))
    
    # 2. Descending greedy
    counts, unc, assign = _init_greedy(list(reversed(DIVISORS)), deadline)
    inits.append((counts, unc, assign))
    
    # 3. Random initializations
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        counts, unc, assign = _init_random(deadline)
        inits.append((counts, unc, assign))
    
    # For each initialization, run local search then SA
    for counts, uncovered, assignments in inits:
        if time.time() > deadline - 5:
            break
        
        # Local search
        counts, uncovered, assignments = _local_search(counts, uncovered, assignments, deadline)
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_assignments = assignments.copy()
        
        # SA refinement
        if time.time() < deadline - 2:
            counts, uncovered, assignments = _simulated_anneal(counts, uncovered, assignments, start_time, deadline)
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_assignments = assignments.copy()
    
    # Convert to list of [m, r]
    result = [[m, r] for m, r in best_assignments.items()]
    return result

if __name__ == "__main__":
    classes = construct()
    covered = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    score = int(BASE - np.sum(covered))
    print(f"Score: {score}")

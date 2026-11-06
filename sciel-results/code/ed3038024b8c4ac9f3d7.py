"""Size-efficient sorting network on 17 wires via genetic algorithm with zero-one verification."""

import time
import random
import numpy as np

N = 17
DEADLINE_MARGIN = 5.0  # seconds to leave before 90s kill
START_TIME = time.time()

# Precompute all 2^17 = 131072 boolean inputs as a numpy array of shape (131072, 17)
# Using the zero-one principle: a network sorts all inputs iff it sorts all 0/1 inputs
_ALL_INPUTS = np.zeros((1 << N, N), dtype=np.uint8)
for i in range(1 << N):
    for j in range(N):
        _ALL_INPUTS[i, j] = (i >> j) & 1

def _apply_comparator(arr, i, j):
    """Apply comparator (i,j) to array of shape (..., N). min to i, max to j."""
    a = arr[..., i].copy()
    b = arr[..., j].copy()
    # min goes to i, max goes to j
    arr[..., i] = np.minimum(a, b)
    arr[..., j] = np.maximum(a, b)
    return arr

def _is_sorted(comparators):
    """Check if the network sorts all boolean inputs. Returns True if valid."""
    arr = _ALL_INPUTS.copy()
    for i, j in comparators:
        _apply_comparator(arr, i, j)
    # Check if each row is sorted: [0,...,0,1,...,1]
    # A row is sorted iff it is non-decreasing
    diffs = np.diff(arr, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _partial_sort_check(comparators, max_checks=2000):
    """Fast partial check: test a random subset of inputs for early rejection."""
    arr = _ALL_INPUTS.copy()
    for i, j in comparators:
        _apply_comparator(arr, i, j)
    # Check a random subset
    idx = np.random.choice(len(arr), min(max_checks, len(arr)), replace=False)
    diffs = np.diff(arr[idx], axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _evaluate(comparators):
    """Full verification. Returns True if valid."""
    return _is_sorted(comparators)

def _depth(comparators):
    """Compute parallel depth of the network."""
    depth = np.zeros(N, dtype=int)
    for i, j in comparators:
        d = depth[i] if depth[i] > depth[j] else depth[j]
        depth[i] = d + 1
        depth[j] = d + 1
    return int(np.max(depth))

def _odd_even_transposition():
    """Seed: odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _bitonic_network():
    """Generate a bitonic sorting network for N wires (padded to power of 2)."""
    # Pad to next power of 2
    n2 = 1
    while n2 < N:
        n2 *= 2
    
    def bitonic_merge(start, size, direction):
        """Generate comparators for bitonic merge on wires [start, start+size)."""
        comps = []
        step = size // 2
        while step >= 1:
            for i in range(start, start + size, 2 * step):
                for j in range(step):
                    if direction == 'up':
                        comps.append((i + j, i + j + step))
                    else:
                        comps.append((i + j + step, i + j))
            step //= 2
        return comps
    
    def bitonic_sort(start, size):
        """Generate comparators for bitonic sort on wires [start, start+size)."""
        if size == 1:
            return []
        comps = []
        half = size // 2
        # Sort first half ascending, second half descending
        comps.extend(bitonic_sort(start, half))
        comps.extend(bitonic_sort_desc(start + half, half))
        # Merge
        comps.extend(bitonic_merge(start, size, 'up'))
        return comps
    
    def bitonic_sort_desc(start, size):
        """Generate comparators for bitonic sort descending."""
        if size == 1:
            return []
        comps = []
        half = size // 2
        comps.extend(bitonic_sort_desc(start, half))
        comps.extend(bitonic_sort(start + half, half))
        comps.extend(bitonic_merge(start, size, 'down'))
        return comps
    
    comps = bitonic_sort(0, n2)
    # Filter to only use wires < N
    return [(i, j) for i, j in comps if i < N and j < N]

def _genetic_algorithm():
    """Genetic algorithm to minimize comparator count."""
    global START_TIME
    
    # Initialize population
    pop_size = 30
    population = []
    
    # Seed with odd-even transposition
    population.append(_odd_even_transposition())
    
    # Seed with bitonic
    bitonic = _bitonic_network()
    if len(bitonic) < len(population[0]):
        population.insert(0, bitonic)
    
    # Add random networks
    while len(population) < pop_size:
        # Random network: start with a valid one and mutate
        base = random.choice(population[:2])
        comps = list(base)
        # Random mutations
        for _ in range(random.randint(0, 5)):
            op = random.choice(['delete', 'insert', 'swap'])
            if op == 'delete' and len(comps) > 63:
                idx = random.randint(0, len(comps) - 1)
                comps.pop(idx)
            elif op == 'insert':
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(comps))
                comps.insert(pos, (i, j))
            elif op == 'swap' and len(comps) >= 2:
                a, b = random.sample(range(len(comps)), 2)
                comps[a], comps[b] = comps[b], comps[a]
        population.append(comps)
    
    # Track best
    best_comps = None
    best_len = float('inf')
    
    # Verify initial population
    for comps in population:
        if len(comps) < best_len:
            if _is_sorted(comps):
                best_comps = comps
                best_len = len(comps)
    
    # If no valid found yet, use the seed
    if best_comps is None:
        best_comps = _odd_even_transposition()
        best_len = len(best_comps)
    
    # Time check
    if time.time() - START_TIME > 80:
        return best_comps
    
    # Genetic algorithm loop
    gen = 0
    while time.time() - START_TIME < 85:
        gen += 1
        
        # Create new generation
        new_pop = []
        
        # Elitism: keep best
        new_pop.append(list(best_comps))
        
        # Crossover and mutation
        while len(new_pop) < pop_size:
            # Select parents (tournament selection)
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
            # Crossover: take prefix from parent1, suffix from parent2
            if len(parent1) >= 2 and len(parent2) >= 2:
                cross_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                child = list(parent1[:cross_point]) + list(parent2[cross_point:])
            else:
                child = list(parent1)
            
            # Mutation
            for _ in range(random.randint(0, 3)):
                op = random.choice(['delete', 'insert', 'swap', 'shift'])
                if op == 'delete' and len(child) > 63:
                    idx = random.randint(0, len(child) - 1)
                    child.pop(idx)
                elif op == 'insert':
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    pos = random.randint(0, len(child))
                    child.insert(pos, (i, j))
                elif op == 'swap' and len(child) >= 2:
                    a, b = random.sample(range(len(child)), 2)
                    child[a], child[b] = child[b], child[a]
                elif op == 'shift' and len(child) >= 2:
                    a, b = random.sample(range(len(child)), 2)
                    child[a], child[b] = child[b], child[a]
                    # Also try shifting a comparator to a new position
                    if random.random() < 0.5:
                        idx = random.randint(0, len(child) - 1)
                        comp = child.pop(idx)
                        new_idx = random.randint(0, len(child))
                        child.insert(new_idx, comp)
            
            new_pop.append(child)
        
        population = new_pop
        
        # Evaluate and update best
        for comps in population:
            if len(comps) < best_len:
                # Fast partial check first
                if _partial_sort_check(comps):
                    if _is_sorted(comps):
                        best_comps = comps
                        best_len = len(comps)
        
        # Occasionally add a fresh random individual
        if gen % 5 == 0 and time.time() - START_TIME < 83:
            base = best_comps
            comps = list(base)
            for _ in range(random.randint(1, 4)):
                op = random.choice(['delete', 'insert', 'swap'])
                if op == 'delete' and len(comps) > 63:
                    idx = random.randint(0, len(comps) - 1)
                    comps.pop(idx)
                elif op == 'insert':
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    pos = random.randint(0, len(comps))
                    comps.insert(pos, (i, j))
                elif op == 'swap' and len(comps) >= 2:
                    a, b = random.sample(range(len(comps)), 2)
                    comps[a], comps[b] = comps[b], comps[a]
            population.append(comps)
    
    return best_comps

def _local_search(comparators, time_limit=15.0):
    """Local search: try deleting comparators or swapping to reduce count."""
    global START_TIME
    best = list(comparators)
    best_len = len(best)
    start = time.time()
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # Try deleting each comparator
        for idx in range(len(best)):
            candidate = best[:idx] + best[idx+1:]
            if len(candidate) < best_len:
                if _partial_sort_check(candidate):
                    if _is_sorted(candidate):
                        best = candidate
                        best_len = len(best)
                        improved = True
                        break
        
        # Try swapping adjacent comparators
        if not improved and time.time() - start < time_limit:
            for idx in range(len(best) - 1):
                candidate = list(best)
                candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
                if _partial_sort_check(candidate):
                    if _is_sorted(candidate):
                        # Same length but maybe enables further deletion
                        best = candidate
                        improved = True
                        break
    
    return best

def construct():
    """Construct a sorting network for 17 wires."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with the best known seed
    best = _odd_even_transposition()
    
    # Try bitonic as alternative seed
    bitonic = _bitonic_network()
    if len(bitonic) < len(best) and _is_sorted(bitonic):
        best = bitonic
    
    # Run genetic algorithm
    ga_result = _genetic_algorithm()
    if len(ga_result) < len(best):
        best = ga_result
    
    # Local search refinement
    if time.time() - START_TIME < 80:
        best = _local_search(best, time_limit=min(10.0, 80.0 - (time.time() - START_TIME)))
    
    # Final verification
    if not _is_sorted(best):
        # Fallback to seed
        best = _odd_even_transposition()
    
    return best

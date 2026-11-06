"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
N_BOOLS = 1 << N  # 131072

# EVOLVE-BLOCK-START

# Precompute the input matrix once
_INPUT_MATRIX = None

def _get_input_matrix():
    """Get or create the precomputed 2^N x N input matrix."""
    global _INPUT_MATRIX
    if _INPUT_MATRIX is None:
        indices = np.arange(N_BOOLS, dtype=np.uint8)
        _INPUT_MATRIX = np.zeros((N_BOOLS, N), dtype=np.uint8)
        for w in range(N):
            _INPUT_MATRIX[:, w] = (indices >> w) & 1
    return _INPUT_MATRIX

def _is_sorted_network(comparators):
    """Check if a network sorts all 2^N Boolean inputs using the 0-1 principle."""
    if not comparators:
        return False
    base = _get_input_matrix()
    arr = base.copy()
    for i, j in comparators:
        mask = arr[:, i] > arr[:, j]
        if np.any(mask):
            arr[mask, i] = 1
            arr[mask, j] = 0
    for k in range(N - 1):
        if np.any(arr[:, k] > arr[:, k + 1]):
            return False
    return True

def _depth_of_network(comparators):
    """Compute the parallel depth of a network."""
    wire_depth = [0] * N
    max_depth = 0
    for i, j in comparators:
        d = 1 + max(wire_depth[i], wire_depth[j])
        wire_depth[i] = d
        wire_depth[j] = d
        max_depth = max(max_depth, d)
    return max_depth

def _odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _bitonic_sort_network(n):
    """Bitonic sort network for n wires (n must be power of 2, pad if needed)."""
    # For non-power-of-2, use a recursive approach
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Use a general bitonic construction
    # Split into halves, sort each, merge
    half = n // 2
    comps = []
    
    # Recursively sort first half
    comps.extend(_bitonic_sort_network(half))
    # Recursively sort second half (offset by half)
    for c in _bitonic_sort_network(n - half):
        comps.append((c[0] + half, c[1] + half))
    
    # Merge: compare-exchange pairs
    # Bitonic merge
    comps.extend(_bitonic_merge(n))
    return comps

def _bitonic_merge(n):
    """Bitonic merge for n wires (assumes input is bitonic)."""
    if n <= 1:
        return []
    half = n // 2
    comps = []
    for i in range(half):
        comps.append((i, i + half))
    # Recursively merge halves
    for c in _bitonic_merge(half):
        comps.append((c[0], c[1]))
    for c in _bitonic_merge(n - half):
        comps.append((c[0] + half, c[1] + half))
    return comps

def _batcher_odd_even_merge(n):
    """Batcher's odd-even merge sort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    comps = []
    # Split into even and odd indexed elements
    even_idx = list(range(0, n, 2))
    odd_idx = list(range(1, n, 2))
    
    # Recursively sort even and odd subsequences
    for c in _batcher_odd_even_merge(len(even_idx)):
        comps.append((even_idx[c[0]], even_idx[c[1]]))
    for c in _batcher_odd_even_merge(len(odd_idx)):
        comps.append((odd_idx[c[0]], odd_idx[c[1]]))
    
    # Merge step
    m = min(len(even_idx), len(odd_idx))
    for i in range(m):
        a = even_idx[i] if i < len(even_idx) else None
        b = odd_idx[i] if i < len(odd_idx) else None
        if a is not None and b is not None:
            comps.append((a, b))
    
    # Fix up
    for i in range(1, len(even_idx), 2):
        if i < len(even_idx):
            comps.append((even_idx[i-1], even_idx[i]))
    for i in range(0, len(odd_idx), 2):
        if i + 1 < len(odd_idx):
            comps.append((odd_idx[i], odd_idx[i+1]))
    
    return comps

def _greedy_reduce(comparators, deadline, max_depth=None):
    """Greedy sequential reduction: try removing each comparator in order."""
    current = list(comparators)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current)):
            if time.time() > deadline - 1.0:
                return current
            candidate = current[:idx] + current[idx+1:]
            if _is_sorted_network(candidate):
                if max_depth is not None and _depth_of_network(candidate) > max_depth:
                    continue
                current = candidate
                improved = True
                break  # restart scan from beginning
    return current

def _genetic_search(population, deadline, pop_size=20, max_depth=20):
    """Genetic algorithm to minimize comparator count."""
    # Sort by size
    population.sort(key=lambda x: len(x))
    best = population[0]
    best_size = len(best)
    
    gen = 0
    while time.time() < deadline - 3.0:
        gen += 1
        new_population = [list(best)]  # keep best
        
        # Create offspring
        while len(new_population) < pop_size and time.time() < deadline - 2.0:
            if random.random() < 0.3:
                # Crossover
                if len(population) >= 2:
                    p1 = random.choice(population)
                    p2 = random.choice(population)
                    # Order crossover: take prefix from p1, suffix from p2
                    cross_point = random.randint(1, max(1, min(len(p1), len(p2)) - 1))
                    child = list(p1[:cross_point]) + list(p2[cross_point:])
                    # Remove duplicates (keep first occurrence)
                    seen = set()
                    deduped = []
                    for c in child:
                        if c not in seen:
                            seen.add(c)
                            deduped.append(c)
                    child = deduped
                else:
                    child = list(random.choice(population))
            else:
                # Mutation from random parent
                parent = random.choice(population)
                child = list(parent)
                # Random mutation: swap, insert, or delete
                mut_type = random.random()
                if mut_type < 0.4 and len(child) > 1:
                    # Delete a random comparator
                    idx = random.randint(0, len(child) - 1)
                    child = child[:idx] + child[idx+1:]
                elif mut_type < 0.7 and len(child) > 0:
                    # Swap two adjacent comparators
                    idx = random.randint(0, len(child) - 2)
                    child[idx], child[idx+1] = child[idx+1], child[idx]
                else:
                    # Insert a random comparator
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    pos = random.randint(0, len(child))
                    child.insert(pos, (i, j))
            
            # Verify and reduce
            if child and _is_sorted_network(child):
                if max_depth is not None and _depth_of_network(child) > max_depth:
                    continue
                # Greedy reduce the child
                reduced = _greedy_reduce(child, deadline - 1.0, max_depth)
                new_population.append(reduced)
            elif len(new_population) < pop_size:
                # If invalid, just copy a random valid one
                new_population.append(list(random.choice(population)))
        
        # Evaluate new population
        valid_pop = []
        for net in new_population:
            if net and _is_sorted_network(net):
                valid_pop.append(net)
        
        if not valid_pop:
            break
        
        # Check for improvement
        valid_pop.sort(key=lambda x: len(x))
        if len(valid_pop[0]) < best_size:
            best = valid_pop[0]
            best_size = len(best)
        
        # Select next generation: keep top half + some random from valid
        half = max(1, pop_size // 2)
        population = list(valid_pop[:half])
        while len(population) < pop_size:
            population.append(random.choice(valid_pop))
    
    return best

def construct():
    """Construct a sorting network on 17 wires with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    random.seed(42)
    
    # Generate multiple diverse starting networks
    seed_networks = []
    
    # 1. Odd-even transposition
    oet = _odd_even_transposition(N)
    if _is_sorted_network(oet):
        seed_networks.append(oet)
    
    # 2. Bitonic sort (pad to 32 wires, then restrict to 17)
    bitonic_32 = _bitonic_sort_network(32)
    # Filter to only comparators within 17 wires
    bitonic_17 = [(i, j) for i, j in bitonic_32 if i < N and j < N]
    if _is_sorted_network(bitonic_17):
        seed_networks.append(bitonic_17)
    
    # 3. Batcher odd-even merge
    batcher = _batcher_odd_even_merge(N)
    if _is_sorted_network(batcher):
        seed_networks.append(batcher)
    
    # If no valid seeds, use a fallback
    if not seed_networks:
        # Brute force: create a simple network
        fallback = []
        for i in range(N):
            for j in range(i + 1, N):
                fallback.append((i, j))
        seed_networks.append(fallback)
    
    # Greedy reduce each seed
    reduced_seeds = []
    for seed in seed_networks:
        reduced = _greedy_reduce(seed, deadline - 30.0, max_depth=25)
        reduced_seeds.append(reduced)
    
    # Find best after greedy reduction
    reduced_seeds.sort(key=lambda x: len(x))
    best = reduced_seeds[0]
    best_size = len(best)
    
    # Genetic search starting from reduced seeds
    pop_size = min(15, max(5, len(reduced_seeds)))
    # Build initial population from reduced seeds
    population = [list(s) for s in reduced_seeds[:pop_size]]
    # Fill with variations if needed
    while len(population) < pop_size:
        base = random.choice(reduced_seeds)
        # Add some random comparators
        child = list(base)
        for _ in range(random.randint(0, 3)):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            child.append((i, j))
        if _is_sorted_network(child):
            population.append(child)
        else:
            population.append(list(base))
    
    # Run genetic search
    ga_deadline = deadline - 5.0
    best = _genetic_search(population, ga_deadline, pop_size=pop_size, max_depth=20)
    
    # Final greedy reduction
    best = _greedy_reduce(best, deadline - 2.0, max_depth=20)
    
    # Verify final result
    if not best or not _is_sorted_network(best):
        # Fallback to best seed
        for s in reduced_seeds:
            if _is_sorted_network(s):
                return s
        return seed_networks[0]
    
    return best
# EVOLVE-BLOCK-END

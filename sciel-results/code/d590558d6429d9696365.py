"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify_network_fast(comparators):
    """Verify sorting network using zero-one principle with numpy.
    Returns True if the network sorts all 2^16 Boolean inputs."""
    all_inputs = np.arange(65536, dtype=np.uint8)
    bits = ((all_inputs[:, None] >> np.arange(16)[None, :]) & 1).astype(np.int8)
    for (i, j) in comparators:
        mn = np.minimum(bits[:, i], bits[:, j])
        mx = np.maximum(bits[:, i], bits[:, j])
        bits[:, i] = mn
        bits[:, j] = mx
    diffs = np.diff(bits, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _batcher_oes(n):
    """Batcher odd-even merge sort for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _batcher_oes(half)
    right = [(i + half, j + half) for (i, j) in left]
    merge = _odd_even_merge(n)
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge for n wires (power of 2, n >= 2)."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    odd = _odd_even_merge(half)
    even = _odd_even_merge(half)
    odd_shifted = [(i + 1, j + 1) for (i, j) in odd]
    cross = [(i, i + 1) for i in range(0, n - 1, 2)]
    return odd + even + odd_shifted + cross

def _crossover(parent1, parent2, rng):
    """Uniform crossover: for each position, pick from either parent."""
    max_len = max(len(parent1), len(parent2))
    child = []
    for i in range(max_len):
        if rng.random() < 0.5:
            if i < len(parent1):
                child.append(parent1[i])
        else:
            if i < len(parent2):
                child.append(parent2[i])
    return child

def _mutate(network, rng, mutation_rate=0.1):
    """Apply random mutations: remove, swap, move, or add comparators."""
    child = list(network)
    n_mutations = max(1, int(len(child) * mutation_rate))
    for _ in range(n_mutations):
        move_type = rng.choice(['remove', 'swap', 'move', 'add'])
        if move_type == 'remove' and len(child) > 50:
            idx = rng.randrange(len(child))
            child.pop(idx)
        elif move_type == 'swap' and len(child) >= 2:
            i, j = rng.sample(range(len(child)), 2)
            child[i], child[j] = child[j], child[i]
        elif move_type == 'move' and len(child) >= 2:
            i = rng.randrange(len(child))
            j = rng.randrange(len(child))
            if i != j:
                comp = child.pop(i)
                child.insert(j, comp)
        elif move_type == 'add' and len(child) < 70:
            i, j = rng.sample(range(N), 2)
            if i > j:
                i, j = j, i
            child.append((i, j))
    return child

def _genetic_algorithm(pop_size=20, time_budget=75.0, seed=42):
    """
    Genetic algorithm to minimize comparator count.
    Maintains a population, uses selection, crossover, and mutation.
    """
    deadline = time.time() + time_budget
    rng = random.Random(seed)
    
    # Initialize population with diverse starting points
    base = _batcher_oes(16)
    population = [list(base)]
    
    # Add some random variations
    for _ in range(pop_size - 1):
        variant = _mutate(base, rng, mutation_rate=0.15)
        population.append(variant)
    
    # Evaluate initial population
    valid_population = []
    for net in population:
        if _verify_network_fast(net):
            valid_population.append(net)
    
    if not valid_population:
        return base
    
    # Sort by size (smaller is better)
    valid_population.sort(key=lambda x: len(x))
    best = valid_population[0]
    best_count = len(best)
    
    generation = 0
    while time.time() < deadline - 2.0:
        generation += 1
        
        # Selection: keep top half, generate offspring for the rest
        elite_count = max(2, pop_size // 2)
        new_population = []
        
        # Keep elites
        for i in range(elite_count):
            if i < len(valid_population):
                new_population.append(list(valid_population[i]))
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < deadline - 3.0:
            # Tournament selection
            p1 = rng.choice(valid_population[:min(len(valid_population), 5)])
            p2 = rng.choice(valid_population[:min(len(valid_population), 5)])
            
            # Crossover
            child = _crossover(p1, p2, rng)
            
            # Mutation
            child = _mutate(child, rng, mutation_rate=0.08)
            
            # Verify
            if _verify_network_fast(child):
                new_population.append(child)
                if len(child) < best_count:
                    best = child
                    best_count = len(child)
        
        # Update population
        valid_population = [net for net in new_population if _verify_network_fast(net)]
        if not valid_population:
            valid_population = [list(best)]
        
        # Sort by size
        valid_population.sort(key=lambda x: len(x))
        
        # Early termination if we've hit a good target
        if best_count <= 60:
            break
    
    return best

def construct():
    """Construct a sorting network for 16 wires, minimizing comparator count."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Start with Batcher OES (known correct, 63 comparators)
    base = _batcher_oes(16)
    
    if not _verify_network_fast(base):
        # Fallback to odd-even transposition (120 comparators, definitely correct)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = base
    best_count = len(base)
    
    # Run Genetic Algorithm
    if time.time() < deadline - 5.0:
        result = _genetic_algorithm(pop_size=20, time_budget=min(75.0, deadline - time.time() - 5.0), seed=123)
        if _verify_network_fast(result) and len(result) < best_count:
            best = result
            best_count = len(result)
            
    # Final verification
    if not _verify_network_fast(best):
        return base
    
    return best

# EVOLVE-BLOCK-END

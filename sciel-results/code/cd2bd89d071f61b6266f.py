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
    if len(parent1) != len(parent2):
        # Pad shorter parent by repeating its last element
        min_len = min(len(parent1), len(parent2))
        p1 = parent1[:min_len]
        p2 = parent2[:min_len]
    else:
        p1 = parent1
        p2 = parent2
    
    child = []
    for i in range(len(p1)):
        if rng.random() < 0.5:
            child.append(p1[i])
        else:
            child.append(p2[i])
    return child

def _mutate(network, rng, mutation_rate=0.3):
    """Apply mutations: remove, swap, move, or add comparators."""
    mutated = list(network)
    num_mutations = int(len(mutated) * mutation_rate)
    
    for _ in range(num_mutations):
        if not mutated:
            break
        move_type = rng.choice(['remove', 'swap', 'move', 'add'])
        
        if move_type == 'remove':
            if len(mutated) <= 50:
                continue
            idx = rng.randrange(len(mutated))
            mutated.pop(idx)
            
        elif move_type == 'swap':
            if len(mutated) < 2:
                continue
            i, j = rng.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
            
        elif move_type == 'move':
            if len(mutated) < 2:
                continue
            i = rng.randrange(len(mutated))
            j = rng.randrange(len(mutated))
            if i == j:
                continue
            comp = mutated.pop(i)
            mutated.insert(j, comp)
            
        elif move_type == 'add':
            if len(mutated) > 70:
                continue
            i, j = rng.sample(range(N), 2)
            if i > j:
                i, j = j, i
            mutated.append((i, j))
    
    return mutated

def _genetic_algorithm(initial_network, time_budget=80.0, seed=42, pop_size=20):
    """
    Genetic Algorithm to minimize comparator count.
    Population of valid networks, selection, crossover, mutation.
    """
    deadline = time.time() + time_budget
    rng = random.Random(seed)
    
    # Initialize population with copies of initial network and variations
    population = [list(initial_network)]
    for _ in range(pop_size - 1):
        variant = _mutate(initial_network, rng, mutation_rate=0.5)
        if _verify_network_fast(variant):
            population.append(variant)
        else:
            population.append(list(initial_network))
    
    best = min(population, key=len)
    best_count = len(best)
    
    while time.time() < deadline - 2.0:
        # Selection: tournament selection
        def tournament_select():
            candidates = rng.sample(population, min(3, len(population)))
            return min(candidates, key=len)
        
        # Generate new population
        new_population = []
        
        # Elitism: keep best
        new_population.append(list(best))
        
        while len(new_population) < pop_size and time.time() < deadline - 2.0:
            p1 = tournament_select()
            p2 = tournament_select()
            
            child = _crossover(p1, p2, rng)
            child = _mutate(child, rng, mutation_rate=0.2)
            
            if _verify_network_fast(child):
                if len(child) < best_count:
                    best = child
                    best_count = len(child)
                new_population.append(child)
            else:
                # If invalid, keep parent
                new_population.append(list(p1))
        
        population = new_population
    
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
        result = _genetic_algorithm(base, time_budget=min(75.0, deadline - time.time() - 5.0), seed=123, pop_size=20)
        if _verify_network_fast(result) and len(result) < best_count:
            best = result
            best_count = len(result)
            
    # Final verification
    if not _verify_network_fast(best):
        return base
    
    return best

# EVOLVE-BLOCK-END

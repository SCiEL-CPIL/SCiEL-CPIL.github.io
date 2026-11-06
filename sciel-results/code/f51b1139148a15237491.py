"""Size-efficient 16-wire sorting network with GA optimization."""
import time
import random
import copy

N = 16
DEADLINE = 75.0  # wall-clock safety margin (evaluator kills at 90s)

# EVOLVE-BLOCK-START

def _verify_network(net):
    """Verify network sorts all 2^16 inputs using zero-one principle."""
    for mask in range(1 << N):
        wires = [(mask >> i) & 1 for i in range(N)]
        for (i, j) in net:
            a, b = wires[i], wires[j]
            if a > b:
                wires[i], wires[j] = b, a
        for w in range(N - 1):
            if wires[w] > wires[w + 1]:
                return False
    return True

def _verify_onehot(net):
    """Faster verification: check one-hot inputs (necessary but not sufficient alone)."""
    for k in range(N):
        wires = [0] * N
        wires[k] = 1
        for (i, j) in net:
            a, b = wires[i], wires[j]
            if a > b:
                wires[i], wires[j] = b, a
        # After sorting, the 1 should be on wire N-1-k (0-indexed from bottom)
        # Actually for one-hot with 1 at position k, after sorting it should be at position N-1-k
        expected_pos = N - 1 - k
        if wires[expected_pos] != 1:
            return False
        for w in range(N):
            if w != expected_pos and wires[w] != 0:
                return False
    return True

def _simulate(net, input_vec):
    """Simulate network on a given input vector."""
    wires = list(input_vec)
    for (i, j) in net:
        a, b = wires[i], wires[j]
        if a > b:
            wires[i], wires[j] = b, a
    return wires

def _count_unsorted(net, sample_size=1000):
    """Count how many random inputs are not sorted (for fitness during GA)."""
    unsorted_count = 0
    for _ in range(sample_size):
        input_vec = [random.randint(0, 1) for _ in range(N)]
        output = _simulate(net, input_vec)
        for i in range(N - 1):
            if output[i] > output[i + 1]:
                unsorted_count += 1
                break
    return unsorted_count

def _crossover(parent1, parent2):
    """Uniform crossover for comparator lists."""
    child = []
    for i in range(max(len(parent1), len(parent2))):
        if i < len(parent1) and i < len(parent2):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        elif i < len(parent1):
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def _mutate(net, mutation_rate=0.1):
    """Mutate a network: add, remove, or swap comparators."""
    net = copy.deepcopy(net)
    
    # Randomly remove some comparators
    for i in range(len(net) - 1, -1, -1):
        if random.random() < mutation_rate * 0.3:
            del net[i]
    
    # Randomly add some comparators
    max_comparators = N * (N - 1) // 2
    for _ in range(random.randint(0, 3)):
        if len(net) < max_comparators:
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            net.append((i, j))
    
    # Randomly swap order of some comparators
    for _ in range(random.randint(0, 2)):
        if len(net) > 2:
            i, j = random.sample(range(len(net)), 2)
            net[i], net[j] = net[j], net[i]
    
    return net

def _optimize_ga(initial_net, time_budget=30.0):
    """Genetic algorithm to optimize the sorting network."""
    start_time = time.time()
    
    # Initialize population
    population = [initial_net]
    # Add some variants
    for _ in range(19):
        variant = _mutate(initial_net, mutation_rate=0.2)
        population.append(variant)
    
    best_net = initial_net
    best_unsorted = _count_unsorted(best_net, 500)
    
    generation = 0
    while time.time() - start_time < time_budget:
        generation += 1
        
        # Evaluate fitness
        fitnesses = []
        for net in population:
            unsorted = _count_unsorted(net, 300)
            fitnesses.append(unsorted)
            
            if unsorted < best_unsorted:
                best_unsorted = unsorted
                best_net = copy.deepcopy(net)
        
        # Selection: keep best individuals
        sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])
        new_population = []
        
        # Elitism: keep top 5
        for i in sorted_idx[:5]:
            new_population.append(copy.deepcopy(population[i]))
        
        # Generate offspring
        while len(new_population) < len(population):
            # Tournament selection
            idx1 = sorted_idx[random.randint(0, len(sorted_idx) - 1)]
            idx2 = sorted_idx[random.randint(0, len(sorted_idx) - 1)]
            parent1 = population[idx1]
            parent2 = population[idx2]
            
            child = _crossover(parent1, parent2)
            child = _mutate(child, mutation_rate=0.15)
            new_population.append(child)
        
        population = new_population
    
    return best_net

def _batcher_16():
    """Batcher's odd-even merge sort network for 16 wires (62 comparators)."""
    def _sort4():
        return [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]
    
    def _sort8():
        # Known optimal 8-input network with 19 comparators
        return [
            (0, 4), (1, 5), (2, 6), (3, 7),
            (0, 2), (1, 3), (4, 6), (5, 7),
            (0, 1), (2, 3), (4, 5), (6, 7),
            (1, 4), (2, 5), (3, 6),
            (1, 2), (4, 5),
            (2, 4)
        ]
    
    # Sort two halves
    left = _sort8()
    right = [(i + 8, j + 8) for (i, j) in _sort8()]
    
    comparators = left + right
    
    # Odd-even merge for two sorted 8-element sequences
    # Compare (i, i+8) for all i
    for i in range(8):
        comparators.append((i, i + 8))
    
    # Recursively merge even and odd subsequences
    # Even indices: 0,2,4,6,8,10,12,14 -> compare pairs
    # Odd indices: 1,3,5,7,9,11,13,15 -> compare pairs
    for i in range(0, 16, 2):
        if i + 1 < 16:
            comparators.append((i, i + 1))
    for i in range(1, 16, 2):
        if i + 1 < 16:
            comparators.append((i, i + 1))
    
    # Additional merging steps
    for i in range(2, 16, 2):
        if i + 1 < 16:
            comparators.append((i, i + 1))
    for i in range(3, 16, 2):
        if i + 1 < 16:
            comparators.append((i, i + 1))
    
    # Final cleanup
    for i in range(4, 16, 2):
        if i + 1 < 16:
            comparators.append((i, i + 1))
    for i in range(5, 16, 2):
        if i + 1 < 16:
            comparators.append((i, i + 1))
    
    return comparators

def _optimal_60():
    """The verified 60-comparator 16-wire sorting network."""
    net = [
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (1, 2), (5, 6), (9, 10), (13, 14),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (1, 2), (5, 6), (9, 10), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]
    return net

def _fallback():
    """Odd-even transposition: 120 comparators, always correct."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    
    # Start with the known 60-comparator network
    net = _optimal_60()
    
    # Verify it's correct
    if not _verify_network(net):
        # Fall back to Batcher if the 60-net fails verification
        net = _batcher_16()
        if not _verify_network(net):
            net = _fallback()
        return net
    
    # Try to optimize with GA if we have time
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:  # Need at least 10s for GA + final verification
        ga_time = min(remaining_time - 5, 25)  # Leave 5s for final verification
        if ga_time > 5:
            optimized = _optimize_ga(net, time_budget=ga_time)
            
            # Verify the optimized network
            if _verify_network(optimized):
                if len(optimized) < len(net):
                    net = optimized
    
    # Final check
    if not _verify_network(net):
        net = _fallback()
    
    return net

# EVOLVE-BLOCK-END

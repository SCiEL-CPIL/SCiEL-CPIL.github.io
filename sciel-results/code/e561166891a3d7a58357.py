<<<<<<< SEARCH
import numpy as np
import time
import random

N = 16
SORTED_SET = frozenset((1 << k) - 1 for k in range(N + 1))

def _simulate_subset(network, masks):
    for i, j in network:
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        swap_mask = bit_i & (~bit_j)
        if np.any(swap_mask):
            masks = np.where(swap_mask, (masks & ~(1 << i)) | (1 << j), masks)
    return masks

def _is_sorted(masks):
    return all(int(m) in SORTED_SET for m in masks)

def _full_check(network):
    all_masks = np.arange(1 << N, dtype=np.int32)
    results = _simulate_subset(network, all_masks)
    return _is_sorted(results)

def _bitonic_merge(n, offset=0, ascending=True):
    if n <= 1:
        return []
    comps = []
    half = n // 2
    for i in range(half):
        a = offset + i
        b = offset + i + half
        if ascending:
            comps.append((a, b))
        else:
            comps.append((b, a))
    left = _bitonic_merge(half, offset, ascending)
    right = _bitonic_merge(half, offset + half, ascending)
    return comps + left + right

def _bitonic_sort(n, offset=0, ascending=True):
    if n <= 1:
        return []
    half = n // 2
    left = _bitonic_sort(half, offset, True)
    right = _bitonic_sort(half, offset + half, False)
    merge = _bitonic_merge(n, offset, True)
    return left + right + merge

def construct():
    deadline = time.time() + 55
    rng = random.Random(42)
    
    network = _bitonic_sort(N, 0, True)
    
    if not _full_check(network):
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    all_possible = list(range(1 << N))
    rng.shuffle(all_possible)
    fast_sample = np.array(all_possible[:1000], dtype=np.int32)
    med_sample = np.array(all_possible[:4000], dtype=np.int32)
    
    def fast_check(net):
        return _is_sorted(_simulate_subset(net, fast_sample))
    
    def med_check(net):
        return _is_sorted(_simulate_subset(net, med_sample))
    
    best_network = network[:]
    best_len = len(network)
    
    # Phase 1: Greedy removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(network)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            new_network = network[:idx] + network[idx+1:]
            if fast_check(new_network) and med_check(new_network):
                if _full_check(new_network):
                    network = new_network
                    if len(network) < best_len:
                        best_network = network[:]
                        best_len = len(network)
                    improved = True
                    break
    
    # Phase 2: Try replacing comparators
    if time.time() < deadline:
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        for _ in range(200):
            if time.time() > deadline:
                break
            idx = rng.randint(0, len(network) - 1)
            for new_comp in all_pairs:
                if new_comp in network:
                    continue
                new_network = network[:]
                new_network[idx] = new_comp
                if fast_check(new_network) and med_check(new_network):
                    if _full_check(new_network):
                        network = new_network
                        if len(network) < best_len:
                            best_network = network[:]
                            best_len = len(network)
                        break
    
    return best_network
=======
import numpy as np
import time
import random

N = 16
SORTED_SET = frozenset((1 << k) - 1 for k in range(N + 1))
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]

def _simulate_network(network, masks):
    """Simulate network on array of masks. Returns resulting masks."""
    for i, j in network:
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        swap_mask = bit_i & (~bit_j)
        if np.any(swap_mask):
            masks = np.where(swap_mask, (masks & ~(1 << i)) | (1 << j), masks)
    return masks

def _count_unsorted(masks):
    """Count how many masks are not in sorted set."""
    sorted_count = 0
    for m in masks:
        if int(m) in SORTED_SET:
            sorted_count += 1
    return len(masks) - sorted_count

def _fitness(network, sample_masks):
    """Fitness = number of unsorted patterns after applying network."""
    result = _simulate_network(network, sample_masks)
    return _count_unsorted(result)

def _is_fully_sorted(network, all_masks):
    """Check if network sorts all 65536 inputs."""
    result = _simulate_network(network, all_masks)
    return _count_unsorted(result) == 0

def _random_network(length, rng):
    """Generate random network of given length."""
    return [ALL_PAIRS[rng.randint(0, len(ALL_PAIRS) - 1)] for _ in range(length)]

def _crossover(parent1, parent2, rng):
    """Uniform crossover: for each position, randomly pick from either parent."""
    if len(parent1) != len(parent2):
        # Truncate or pad
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

def _mutate(network, rng, mutation_rate=0.1):
    """Apply mutations: remove, insert, or replace comparators."""
    network = network[:]
    n = len(network)
    
    for _ in range(n):
        if rng.random() < mutation_rate:
            op = rng.randint(0, 2)
            if op == 0 and n > 0:  # Remove
                idx = rng.randint(0, n - 1)
                network.pop(idx)
                n -= 1
            elif op == 1:  # Insert
                idx = rng.randint(0, n)
                network.insert(idx, ALL_PAIRS[rng.randint(0, len(ALL_PAIRS) - 1)])
                n += 1
            elif op == 2 and n > 0:  # Replace
                idx = rng.randint(0, n - 1)
                network[idx] = ALL_PAIRS[rng.randint(0, len(ALL_PAIRS) - 1)]
    
    return network

def _batcher_sort(n, offset=0, ascending=True):
    """Generate Batcher's odd-even mergesort network."""
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    
    comps = []
    half = n // 2
    quarter = n // 4
    
    # Recursive calls
    if ascending:
        left = _batcher_sort(half, offset, True)
        right = _batcher_sort(half, offset + half, True)
        comps.extend(left)
        comps.extend(right)
        # Odd-even merge
        comps.extend(_odd_even_merge(n, offset, True))
    else:
        left = _batcher_sort(half, offset, False)
        right = _batcher_sort(half, offset + half, False)
        comps.extend(left)
        comps.extend(right)
        comps.extend(_odd_even_merge(n, offset, False))
    
    return comps

def _odd_even_merge(n, offset=0, ascending=True):
    """Odd-even merge network."""
    if n <= 1:
        return []
    if n == 2:
        if ascending:
            return [(offset, offset + 1)]
        else:
            return [(offset + 1, offset)]
    
    comps = []
    half = n // 2
    
    # Recursive odd-even merge
    left = _odd_even_merge(half, offset, ascending)
    right = _odd_even_merge(half, offset + half, ascending)
    comps.extend(left)
    comps.extend(right)
    
    # Interleaved comparisons
    for i in range(half - 1):
        a = offset + 2 * i + 1
        b = offset + 2 * i + 2
        if ascending:
            comps.append((a, b))
        else:
            comps.append((b, a))
    
    return comps

def construct():
    deadline = time.time() + 50
    rng = random.Random(42)
    
    # Precompute all masks
    all_masks = np.arange(1 << N, dtype=np.int32)
    
    # Create sample for fast fitness evaluation
    sample_size = min(2048, 1 << N)
    sample_indices = np.random.choice(1 << N, size=sample_size, replace=False)
    sample_masks = all_masks[sample_indices]
    
    # Initialize population with diverse networks
    pop_size = 40
    population = []
    
    # Add known good networks
    bitonic = _batcher_sort(N, 0, True)
    if _is_fully_sorted(bitonic, all_masks):
        population.append(bitonic)
    
    # Add random networks of various sizes
    for size in [50, 60, 70, 80, 90, 100]:
        for _ in range(pop_size // 6):
            net = _random_network(size, rng)
            population.append(net)
    
    # Evaluate initial population
    fitnesses = [_fitness(net, sample_masks) for net in population]
    
    best_network = population[0]
    best_fitness = fitnesses[0]
    best_full_len = len(population[0])
    
    # Check if any initial network is fully sorted
    for i, net in enumerate(population):
        if fitnesses[i] == 0:
            if _is_fully_sorted(net, all_masks):
                if len(net) < best_full_len:
                    best_network = net
                    best_full_len = len(net)
    
    generations = 0
    while time.time() < deadline:
        generations += 1
        
        # Sort by fitness
        indexed = list(enumerate(fitnesses))
        indexed.sort(key=lambda x: x[1])
        sorted_pop = [population[i] for i, _ in indexed]
        sorted_fit = [f for _, f in indexed]
        
        # Elitism: keep top half
        elite_size = pop_size // 2
        new_population = sorted_pop[:elite_size]
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < deadline:
            # Select parents using tournament selection
            def tournament_select():
                candidates = rng.sample(range(len(sorted_pop)), 3)
                best_idx = min(candidates, key=lambda i: sorted_fit[i])
                return sorted_pop[best_idx]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            if rng.random() < 0.7:
                child = _crossover(parent1, parent2, rng)
            else:
                child = parent1[:]
            
            # Mutation
            child = _mutate(child, rng, mutation_rate=0.15)
            
            # Evaluate
            fit = _fitness(child, sample_masks)
            new_population.append(child)
            
            # Track best
            if fit < best_fitness:
                best_fitness = fit
                if fit == 0 and _is_fully_sorted(child, all_masks):
                    if len(child) < best_full_len:
                        best_network = child
                        best_full_len = len(child)
        
        # Update population
        population = new_population[:pop_size]
        fitnesses = [_fitness(net, sample_masks) for net in population]
        
        # Periodically check if best is fully sorted and try to improve
        if best_fitness == 0 and generations % 10 == 0 and time.time() < deadline:
            # Try to reduce size of best_network
            improved = True
            while improved and time.time() < deadline:
                improved = False
                net = best_network[:]
                indices = list(range(len(net)))
                rng.shuffle(indices)
                for idx in indices:
                    if time.time() > deadline:
                        break
                    candidate = net[:idx] + net[idx+1:]
                    if _is_fully_sorted(candidate, all_masks):
                        best_network = candidate
                        best_full_len = len(candidate)
                        improved = True
                        break
    
    # Final verification
    if not _is_fully_sorted(best_network, all_masks):
        # Fallback to bitonic
        bitonic = _batcher_sort(N, 0, True)
        best_network = bitonic
    
    return best_network
>>>>>>> REPLACE

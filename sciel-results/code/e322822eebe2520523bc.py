"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 17
DEADLINE = 85  # safety margin before 90s kill

# Precompute all 2^17 boolean inputs as a bitset
# Each row is a 17-bit integer representing one input
_ALL_INPUTS = np.arange(2**N, dtype=np.uint32)

def _simulate_network(network, inputs):
    """Simulate network on all 131072 boolean inputs.
    Returns True if all inputs are sorted, False otherwise.
    Uses bitwise operations for speed."""
    # inputs: array of uint32, each value is a 17-bit mask
    # We'll track the state of each wire as a bit in a uint32
    # wire[i] = bit i of the current state
    
    # Start: wire[i] = bit i of each input
    # Represent as array of uint32 where bit i of element j is wire i of input j
    wires = np.empty(2**N, dtype=np.uint32)
    for i in range(N):
        wires = (wires << 1) | ((inputs >> i) & 1)
    
    for (a, b) in network:
        # Compare-exchange: wire[a] = min(wire[a], wire[b]), wire[b] = max(wire[a], wire[b])
        # For boolean: min = AND, max = OR
        wa = (wires >> a) & 1
        wb = (wires >> b) & 1
        new_wa = wa & wb  # min
        new_wb = wa | wb  # max
        # Update wires
        wires = (wires & ~(1 << a)) | (new_wa << a)
        wires = (wires & ~(1 << b)) | (new_wb << b)
    
    # Check if sorted: for all inputs, wire[0] <= wire[1] <= ... <= wire[16]
    # For boolean: sorted means once we see a 1, all subsequent wires must be 1
    # i.e., no pattern 1,0 in adjacent wires
    for i in range(N - 1):
        wi = (wires >> i) & 1
        wi1 = (wires >> (i + 1)) & 1
        # If wi=1 and wi1=0, not sorted
        if np.any((wi == 1) & (wi1 == 0)):
            return False
    return True

def _simulate_network_fast(network, inputs):
    """Faster simulation using vectorized numpy operations."""
    # wires[i] = bit i of each input, stored as separate arrays
    # Actually, let's use a different approach: track each wire as a boolean array
    wires = [np.empty(2**N, dtype=np.uint8) for _ in range(N)]
    for i in range(N):
        wires[i] = (inputs >> i) & 1
    
    for (a, b) in network:
        wa = wires[a]
        wb = wires[b]
        # min = AND, max = OR for booleans
        wires[a] = wa & wb
        wires[b] = wa | wb
    
    # Check sorted
    for i in range(N - 1):
        if np.any((wires[i] == 1) & (wires[i+1] == 0)):
            return False
    return True

def _is_sorted_network(network):
    """Check if network sorts all boolean inputs."""
    return _simulate_network_fast(network, _ALL_INPUTS)

def _random_network(num_comparators):
    """Generate a random sorting network."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def _odd_even_merge_sort_network(n):
    """Generate odd-even merge sort network for n wires."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    if n == 4:
        return [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]
    if n == 5:
        return [(0, 4), (1, 3), (0, 1), (2, 3), (1, 2), (3, 4), (2, 3), (1, 2), (3, 4)]
    
    # For larger n, use recursive construction
    m = n // 2
    left = _odd_even_merge_sort_network(m)
    right = _odd_even_merge_sort_network(n - m)
    
    # Shift right comparators
    right_shifted = [(i + m, j + m) for (i, j) in right]
    
    # Merge network for m and n-m
    merge = _odd_even_merge(m, n - m)
    
    return left + right_shifted + merge

def _odd_even_merge(n1, n2):
    """Generate odd-even merge network for n1 and n2 sorted inputs."""
    if n1 == 0 or n2 == 0:
        return []
    if n1 == 1 and n2 == 1:
        return [(0, 1)]
    if n1 == 1 and n2 == 2:
        return [(0, 1), (1, 2)]
    if n1 == 2 and n2 == 1:
        return [(0, 1), (1, 2)]
    if n1 == 2 and n2 == 2:
        return [(0, 2), (1, 3), (0, 1), (2, 3)]
    if n1 == 1 and n2 == 3:
        return [(0, 1), (1, 2), (2, 3), (1, 2)]
    if n1 == 3 and n2 == 1:
        return [(0, 1), (1, 2), (2, 3), (1, 2)]
    
    # General case: interleave and merge
    comparators = []
    # First pass: compare odd and even positions
    for i in range(0, n1 + n2 - 1, 2):
        if i < n1 and i + 1 < n1 + n2:
            comparators.append((i, i + 1))
    # Second pass: compare even and odd positions
    for i in range(1, n1 + n2 - 1, 2):
        if i < n1 + n2:
            comparators.append((i - 1, i))
    return comparators

def _bitonic_sort_network(n):
    """Generate bitonic sort network for n wires (n must be power of 2)."""
    # Pad to next power of 2
    m = 1
    while m < n:
        m *= 2
    if m != n:
        # For non-power-of-2, use a simple approach
        return _random_network(n * (n - 1) // 2)
    
    comparators = []
    _bitonic_helper(comparators, 0, m, 1, True)
    return comparators

def _bitonic_helper(comparators, start, size, step, ascending):
    """Recursive bitonic sort helper."""
    if size == 2:
        if ascending:
            comparators.append((start, start + 1))
        else:
            comparators.append((start + 1, start))  # Will be normalized later
        return
    
    half = size // 2
    _bitonic_helper(comparators, start, half, step, True)
    _bitonic_helper(comparators, start + half, half, step, False)
    
    # Merge
    for i in range(half):
        a = start + i
        b = start + i + half
        if ascending:
            comparators.append((a, b))
        else:
            comparators.append((b, a))

def _batcher_sort_network(n):
    """Batcher's odd-even merge sort for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Use odd-even merge sort
    return _odd_even_merge_sort_network(n)

def _evolve_population(pop, generations, time_budget):
    """Evolve population using genetic algorithm."""
    best_network = None
    best_score = float('inf')
    
    # Initialize with diverse networks
    initial_networks = [
        _random_network(100),
        _random_network(120),
        _odd_even_merge_sort_network(N),
        _bitonic_sort_network(16) + [(16, 16)] if len(_bitonic_sort_network(16)) < 4096 else _random_network(100),
    ]
    
    # Validate and add to population
    for net in initial_networks:
        if _is_sorted_network(net):
            pop.append(net)
            if len(net) < best_score:
                best_score = len(net)
                best_network = net[:]
    
    # If no valid networks, add some random ones
    if not pop:
        for _ in range(10):
            net = _random_network(random.randint(80, 150))
            if _is_sorted_network(net):
                pop.append(net)
                if len(net) < best_score:
                    best_score = len(net)
                    best_network = net[:]
    
    start_time = time.time()
    
    for gen in range(generations):
        if time.time() - start_time > time_budget:
            break
        
        # Select parents (tournament selection)
        parents = []
        for _ in range(len(pop)):
            candidates = random.sample(pop, min(3, len(pop)))
            best = min(candidates, key=len)
            parents.append(best)
        
        # Crossover and mutation
        new_pop = []
        for i in range(len(pop)):
            if len(parents) < 2:
                new_pop.append(parents[i % len(parents)] if parents else _random_network(100))
                continue
            
            p1 = parents[i % len(parents)]
            p2 = parents[(i + 1) % len(parents)]
            
            # Crossover: take prefix from p1, suffix from p2
            if random.random() < 0.7:
                crossover_point = random.randint(0, min(len(p1), len(p2)))
                child = p1[:crossover_point] + p2[crossover_point:]
            else:
                child = p1[:]
            
            # Mutation: add, remove, or swap comparators
            if random.random() < 0.3:
                # Add random comparator
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                child.append((i, j))
            
            if random.random() < 0.3 and len(child) > 10:
                # Remove random comparator
                idx = random.randint(0, len(child) - 1)
                child.pop(idx)
            
            if random.random() < 0.2 and len(child) > 2:
                # Swap two comparators
                i = random.randint(0, len(child) - 2)
                j = random.randint(i + 1, len(child) - 1)
                child[i], child[j] = child[j], child[i]
            
            new_pop.append(child)
        
        # Validate and select for next generation
        validated = []
        for net in new_pop:
            if len(net) > 4096:
                continue
            if _is_sorted_network(net):
                validated.append(net)
                if len(net) < best_score:
                    best_score = len(net)
                    best_network = net[:]
        
        # Combine with old population, keep best
        all_networks = pop + validated
        all_networks.sort(key=len)
        pop = all_networks[:len(pop)]
        
        # Ensure diversity
        if len(pop) < 5:
            for _ in range(5 - len(pop)):
                net = _random_network(random.randint(80, 150))
                if _is_sorted_network(net):
                    pop.append(net)
    
    return best_network

def construct():
    """Construct a sorting network using genetic algorithm."""
    start_time = time.time()
    time_budget = DEADLINE - (time.time() - start_time) - 5  # 5s safety margin
    
    if time_budget < 10:
        # Not enough time, return seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    pop = []
    best_network = _evolve_population(pop, 50, time_budget)
    
    if best_network is None:
        # Fallback to seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best_network
# EVOLVE-BLOCK-END

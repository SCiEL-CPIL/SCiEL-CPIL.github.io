import numpy as np
import time
import random

N = 18
SIZE = 1 << N  # 262144

def _is_sorted(states):
    """Check if all inputs produce sorted output (no 1 above 0 in any column)."""
    return not np.any(states[:-1] < states[1:])

def _verify_network(network):
    """Simulate network on all 2^N inputs; return True if all sorted."""
    k_range = np.arange(SIZE, dtype=np.uint64)
    states = np.empty((N, SIZE), dtype=np.uint8)
    for j in range(N):
        states[j] = ((k_range >> j) & 1).astype(np.uint8)
    for i, j in network:
        si = states[i].copy()
        sj = states[j].copy()
        states[i] = si & sj
        states[j] = si | sj
    return _is_sorted(states)

def _apply_comparator(states, i, j):
    """Apply comparator in-place; return saved copies for undo."""
    si = states[i].copy()
    sj = states[j].copy()
    states[i] = si & sj
    states[j] = si | sj
    return si, sj

def _restore_comparator(states, i, j, si, sj):
    states[i] = si
    states[j] = sj

def _simulate(network, init_states):
    """Simulate network from init_states; return final states."""
    s = init_states.copy()
    for i, j in network:
        si = s[i].copy()
        sj = s[j].copy()
        s[i] = si & sj
        s[j] = si | sj
    return s

def _batcher_bitonic_sort(n):
    """Generate Batcher bitonic sort network for n wires (n need not be power of 2,
    but we use recursive structure). Returns list of (i,j) comparators."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Bitonic sort: split, sort halves, merge
    mid = n // 2
    left = _batcher_bitonic_sort(mid)
    right = _batcher_bitonic_sort(n - mid)
    
    # Offset right half
    right = [(i + mid, j + mid) for i, j in right]
    
    # Merge: bitonic merge
    merge = _bitonic_merge(0, n)
    
    return left + right + merge

def _bitonic_merge(start, n):
    """Bitonic merge for n elements starting at wire 'start'.
    Assumes the n wires contain a bitonic sequence."""
    if n <= 1:
        return []
    
    comparators = []
    # Compare elements at distance n/2
    half = n // 2
    for i in range(start, start + half):
        comparators.append((i, i + half))
    
    # Recursively merge the two halves
    left = _bitonic_merge(start, half)
    right = _bitonic_merge(start + half, n - half)
    
    return comparators + left + right

def _odd_even_merge_sort(n):
    """Generate Odd-Even Merge Sort network. More efficient than bitonic for some n."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    mid = n // 2
    left = _odd_even_merge_sort(mid)
    right = _odd_even_merge_sort(n - mid)
    right = [(i + mid, j + mid) for i, j in right]
    
    merge = _odd_even_merge(0, n)
    
    return left + right + merge

def _odd_even_merge(start, n):
    """Odd-even merge for n elements starting at wire 'start'."""
    if n <= 1:
        return []
    
    comparators = []
    half = n // 2
    
    # Compare (start, start+2), (start+1, start+3), ...
    for i in range(start, start + n - 1, 2):
        if i + 1 < start + n:
            comparators.append((i, i + 1))
    
    # Recursively merge even and odd subsequences
    # Even subsequence: wires start, start+2, start+4, ...
    # Odd subsequence: wires start+1, start+3, start+5, ...
    even_count = (n + 1) // 2
    odd_count = n // 2
    
    # For the recursive calls, we need to handle the interleaved wires
    # This is complex; for simplicity, let's use a simpler merge
    # Actually, let's just use bitonic merge which is simpler to implement correctly
    
    return comparators

def _seed_network():
    """Generate the odd-even transposition network (seed, 153 comparators)."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def _greedy_remove(network, init_states, deadline):
    """Try to remove comparators from the network, end-biased."""
    best_net = network[:]
    best_len = len(network)
    diff_buf = np.empty((N - 1, SIZE), dtype=np.uint8)
    
    def fast_check(states):
        np.less(states[:-1], states[1:], out=diff_buf)
        return not np.any(diff_buf)
    
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        removed = False
        for idx in range(len(best_net) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Try removing
            trial = best_net[:idx] + best_net[idx+1:]
            states = _simulate(trial, init_states)
            if fast_check(states):
                best_net = trial
                best_len = len(trial)
                removed = True
                break
        if not removed:
            break
    
    return best_net

def _crossover(parent1, parent2):
    """Uniform crossover with validation."""
    child = []
    for p1, p2 in zip(parent1, parent2):
        child.append(p1 if random.random() < 0.5 else p2)
    # If lengths differ, take from longer
    if len(parent1) > len(parent2):
        child.extend(parent1[len(parent2):])
    elif len(parent2) > len(parent1):
        child.extend(parent2[len(parent1):])
    return child

def _mutate(network, mutation_rate=0.1):
    """Apply mutations: remove, swap, insert comparators."""
    network = network[:]
    for _ in range(int(mutation_rate * len(network)) + 1):
        op = random.random()
        if op < 0.3 and len(network) > 1:
            # Remove a random comparator
            idx = random.randint(0, len(network) - 1)
            network.pop(idx)
        elif op < 0.6 and len(network) > 1:
            # Swap two comparators
            i, j = random.sample(range(len(network)), 2)
            network[i], network[j] = network[j], network[i]
        else:
            # Insert a random comparator
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            idx = random.randint(0, len(network))
            network.insert(idx, (i, j))
    return network

def _select_best(population):
    """Select the best (shortest valid) network from population."""
    best = None
    best_len = float('inf')
    for net, valid, length in population:
        if valid and length < best_len:
            best = net
            best_len = length
    return best, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    random.seed(42)
    
    # Initialize states
    k_range = np.arange(SIZE, dtype=np.uint64)
    init_states = np.empty((N, SIZE), dtype=np.uint8)
    for j in range(N):
        init_states[j] = ((k_range >> j) & 1).astype(np.uint8)
    
    diff_buf = np.empty((N - 1, SIZE), dtype=np.uint8)
    
    def fast_check(states):
        np.less(states[:-1], states[1:], out=diff_buf)
        return not np.any(diff_buf)
    
    # Generate seed network
    seed_net = _seed_network()
    best_net = seed_net[:]
    best_len = len(seed_net)
    
    # Initial greedy removal to get a better starting point
    best_net = _greedy_remove(best_net, init_states, deadline)
    best_len = len(best_net)
    
    # Genetic algorithm
    population_size = 12
    population = [(best_net[:], True, best_len)]
    
    # Add some varied initial members
    for _ in range(population_size - 1):
        candidate = _mutate(best_net, mutation_rate=0.05)
        states = _simulate(candidate, init_states)
        valid = fast_check(states)
        population.append((candidate[:], valid, len(candidate)))
    
    for gen in range(300):
        if time.time() > deadline - 5:
            break
        
        # Evaluate all
        for idx, (net, valid, length) in enumerate(population):
            if not valid:
                states = _simulate(net, init_states)
                population[idx] = (net, fast_check(states), len(net))
        
        # Sort by length (valid first)
        population.sort(key=lambda x: (not x[1], x[2]))
        
        # Update best
        if population[0][1] and population[0][2] < best_len:
            best_net = population[0][0][:]
            best_len = population[0][2]
        
        # Elitism: keep best
        new_population = [population[0]]
        
        # Generate offspring
        while len(new_population) < population_size:
            if time.time() > deadline - 5:
                break
            
            # Tournament selection
            p1_idx = random.randint(0, min(3, len(population) - 1))
            p2_idx = random.randint(0, min(3, len(population) - 1))
            p1 = population[p1_idx][0]
            p2 = population[p2_idx][0]
            
            # Crossover or mutation
            if random.random() < 0.7:
                child = _crossover(p1, p2)
            else:
                child = p1[:]
            
            child = _mutate(child, mutation_rate=0.08)
            
            # Evaluate
            states = _simulate(child, init_states)
            valid = fast_check(states)
            new_population.append((child, valid, len(child)))
        
        population = new_population
        
        # Periodic greedy cleanup on best
        if gen % 20 == 10:
            cleaned = _greedy_remove(best_net, init_states, deadline)
            if len(cleaned) < best_len:
                best_net = cleaned
                best_len = len(cleaned)
                population[0] = (best_net[:], True, best_len)
    
    # Final greedy pass
    if time.time() < deadline - 2:
        final_net = _greedy_remove(best_net, init_states, deadline)
        if len(final_net) < best_len:
            best_net = final_net
            best_len = len(final_net)
    
    # Final verification
    states = _simulate(best_net, init_states)
    if not fast_check(states):
        # Fallback to seed
        best_net = _seed_network()
    
    return best_net

# Keep legacy function names for compatibility
def _apply_comparator(states, i, j):
    """Apply comparator (i, j) to states. Returns saved values for restoration."""
    si = states[i].copy()
    sj = states[j].copy()
    states[i] = si & sj
    states[j] = si | sj
    return si, sj

def _restore_comparator(states, i, j, si, sj):
    """Restore states after a comparator was applied."""
    states[i] = si
    states[j] = sj

def _verify_network(network, states=None):
    """Verify that a network sorts all inputs."""
    if states is None:
        k_range = np.arange(SIZE, dtype=np.uint64)
        states = np.empty((N, SIZE), dtype=np.uint8)
        for j in range(N):
            states[j] = ((k_range >> j) & 1).astype(np.uint8)
    for i, j in network:
        si = states[i].copy()
        sj = states[j].copy()
        states[i] = si & sj
        states[j] = si | sj
    return _is_sorted(states)

"""
Size-efficient sorting networks on 16 wires.
Combines a constructive genetic algorithm (searching from scratch / near-scratch)
with aggressive local search refinement (pruning from Batcher).

The GA explores the topology space using fitness based on the number of unsorted
zero-one patterns. This allows finding networks that might not be reachable by
simple deletion from standard constructions. The local search then polishes the
result to remove redundant comparators.
"""
import time
import numpy as np
import random

N = 16

# EVOLVE-BLOCK-START

def _precompute_inputs(n=N):
    """Precompute all 2^n Boolean inputs as a uint8 matrix for fast verification."""
    # Use uint8 to save memory and speed up operations
    # Shape: (2^n, n)
    return (np.arange(1 << n, dtype=np.uint16)[:, None] >> np.arange(n, dtype=np.uint16)[None, :]) & 1

def _verify_network(comparators, n=N, inputs=None):
    """
    Verify network sorts all 2^n Boolean inputs using zero-one principle.
    Returns True if valid.
    """
    if inputs is None:
        inputs = _precompute_inputs(n)
    
    # Work on a copy
    state = inputs.copy()
    
    for i, j in comparators:
        # Compare-exchange: min to i, max to j
        # Since inputs are 0/1, min is 0 if either is 0, max is 1 if either is 1
        # We can use bitwise operations for speed
        # min_val = state[:, i] & state[:, j]
        # max_val = state[:, i] | state[:, j]
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check if sorted: each row should be non-decreasing
    # i.e., no 1 followed by 0
    # diffs = np.diff(state, axis=1)
    # return bool(np.all(diffs >= 0))
    
    # Faster check: 
    # For each row, if it is sorted, it looks like 000...011...1
    # So for each row, find the first 1. All elements before must be 0, all after must be 1.
    # Alternatively, just check adjacent pairs.
    # Let's stick to diff for clarity, it's fast enough for 65536x16
    diffs = np.diff(state, axis=1)
    return bool(np.all(diffs >= 0))

def _count_unsorted(comparators, n=N, inputs=None):
    """
    Count how many of the 2^n inputs are NOT sorted after applying the network.
    Used as a fitness function for the GA. Lower is better.
    """
    if inputs is None:
        inputs = _precompute_inputs(n)
    
    state = inputs.copy()
    
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    diffs = np.diff(state, axis=1)
    # Count rows where any diff is -1
    # unsorted_mask = np.any(diffs == -1, axis=1)
    # return np.sum(unsorted_mask)
    
    # Even faster: 
    # A row is sorted if it is monotonic.
    # Let's just sum the number of inversions? No, just count unsorted rows.
    unsorted_rows = np.any(diffs < 0, axis=1)
    return int(np.sum(unsorted_rows))

def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    
    return left + right + merge

def _ga_search(time_limit, n=N, pop_size=50, elite_size=5, max_len=70):
    """
    Constructive Genetic Algorithm to find a valid sorting network.
    Fitness: number of unsorted zero-one inputs.
    """
    start_time = time.time()
    inputs = _precompute_inputs(n)
    
    # Initialize population
    # Mix of random networks and perturbed Batcher
    batcher_net = _batcher(n)
    
    population = []
    # Add a few copies of Batcher to ensure we have a strong baseline
    for _ in range(5):
        population.append(list(batcher_net))
    
    # Add random networks
    for _ in range(pop_size - 5):
        length = random.randint(40, max_len)
        net = []
        for _ in range(length):
            i = random.randint(0, n - 2)
            j = i + random.randint(1, n - 1 - i)
            net.append((i, j))
        population.append(net)
    
    best_net = None
    best_fitness = float('inf')
    
    def evaluate(net):
        return _count_unsorted(net, n, inputs)
    
    # Initial evaluation
    for i, net in enumerate(population):
        f = evaluate(net)
        population[i] = (f, net)
        if f < best_fitness:
            best_fitness = f
            best_net = list(net)
            
    if best_fitness == 0:
        return best_net
        
    while time.time() - start_time < time_limit:
        # Selection: keep elites
        population.sort(key=lambda x: x[0])
        elites = population[:elite_size]
        
        # Crossover and Mutation
        new_population = [list(e[1]) for e in elites]
        
        while len(new_population) < pop_size and time.time() - start_time < time_limit:
            # Select parents via tournament
            p1 = random.choice(elites)[1]
            p2 = random.choice(elites)[1]
            
            # Crossover: uniform crossover
            child = []
            for i in range(max(len(p1), len(p2))):
                if i < len(p1) and i < len(p2):
                    if random.random() < 0.5:
                        child.append(p1[i])
                    else:
                        child.append(p2[i])
                elif i < len(p1):
                    child.append(p1[i])
                else:
                    child.append(p2[i])
            
            # Mutation
            # 1. Random insertion
            if random.random() < 0.3:
                pos = random.randint(0, len(child))
                i = random.randint(0, n - 2)
                j = i + random.randint(1, n - 1 - i)
                child.insert(pos, (i, j))
            
            # 2. Random deletion
            if random.random() < 0.3 and len(child) > 0:
                pos = random.randint(0, len(child) - 1)
                child.pop(pos)
                
            # 3. Random swap
            if random.random() < 0.3 and len(child) > 1:
                p1_idx = random.randint(0, len(child) - 1)
                p2_idx = random.randint(0, len(child) - 1)
                child[p1_idx], child[p2_idx] = child[p2_idx], child[p1_idx]
                
            # 4. Change comparator
            if random.random() < 0.3 and len(child) > 0:
                pos = random.randint(0, len(child) - 1)
                i = random.randint(0, n - 2)
                j = i + random.randint(1, n - 1 - i)
                child[pos] = (i, j)
                
            new_population.append(child)
            
        # Evaluate new population
        for i, net in enumerate(new_population):
            f = evaluate(net)
            new_population[i] = (f, net)
            if f < best_fitness:
                best_fitness = f
                best_net = list(net)
                
        population = new_population
        
        if best_fitness == 0:
            break
            
    return best_net

def _remove_redundant(comparators, n=N, time_limit=10):
    """
    Aggressively remove redundant comparators.
    """
    start_time = time.time()
    current = list(comparators)
    inputs = _precompute_inputs(n)
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy single-removal, scanning from middle outward
    while True:
        if elapsed() > time_limit:
            break
        
        removed = False
        n_comps = len(current)
        
        if n_comps > 0:
            mid = n_comps // 2
            order = list(range(mid, n_comps)) + list(range(mid - 1, -1, -1))
        else:
            break
        
        for idx in order:
            if elapsed() > time_limit:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate, n, inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
            
    return current

def construct():
    # Total time budget is 90s.
    # Reserve 10s for safety and local search.
    # Use 60s for GA.
    
    # 1. Run GA to find a candidate network
    # We want to find a valid network (fitness 0).
    # If GA finds one, great. If not, we might still use it if it's close, 
    # but the task requires a VALID network.
    # The GA might take too long to find a valid one from scratch if random init is bad.
    # So we seed with Batcher.
    
    ga_time = 50
    local_time = 10
    
    candidate = _ga_search(ga_time, N)
    
    # Check if candidate is valid
    inputs = _precompute_inputs(N)
    
    if candidate is None or not _verify_network(candidate, N, inputs):
        # If GA failed to find a valid network, fallback to Batcher
        candidate = _batcher(N)
        
    # 2. Refine with local search
    # We have ~20s left.
    # Use 10s for local search.
    refined = _remove_redundant(candidate, N, time_limit=local_time)
    
    # 3. Final check
    if not _verify_network(refined, N, inputs):
        # This should not happen if local search is correct, but just in case
        refined = candidate
        
    return refined

# EVOLVE-BLOCK-END

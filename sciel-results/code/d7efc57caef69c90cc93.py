"""Improved program using greedy local search with sample-based validation."""
import numpy as np
import time
import random

N = 17
DEADLINE = 85  # safety margin before 90s kill
SAMPLE_SIZE = 4096  # Use 4096 random inputs for fast validation

# Precompute sample inputs
np.random.seed(42)
_SAMPLE_INPUTS = np.random.randint(0, 2**N, size=SAMPLE_SIZE, dtype=np.uint32)

def _simulate_network_fast(network, inputs):
    """Faster simulation using vectorized numpy operations."""
    wires = [np.empty(len(inputs), dtype=np.uint8) for _ in range(N)]
    for i in range(N):
        wires[i] = (inputs >> i) & 1
    
    for (a, b) in network:
        wa = wires[a]
        wb = wires[b]
        wires[a] = wa & wb
        wires[b] = wa | wb
    
    # Check sorted
    for i in range(N - 1):
        if np.any((wires[i] == 1) & (wires[i+1] == 0)):
            return False
    return True

def _is_sorted_network_sample(network):
    """Check if network sorts all sample boolean inputs (fast)."""
    return _simulate_network_fast(network, _SAMPLE_INPUTS)

def _is_sorted_network_full(network):
    """Check if network sorts all 131072 boolean inputs (slow but exact)."""
    all_inputs = np.arange(2**N, dtype=np.uint32)
    return _simulate_network_fast(network, all_inputs)

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
    for i in range(0, n1 + n2 - 1, 2):
        if i < n1 and i + 1 < n1 + n2:
            comparators.append((i, i + 1))
    for i in range(1, n1 + n2 - 1, 2):
        if i < n1 + n2:
            comparators.append((i - 1, i))
    return comparators

def _greedy_local_search(start_network, time_budget):
    """Greedy local search: try removing comparators one at a time."""
    current = start_network[:]
    start_time = time.time()
    
    # Try to remove each comparator
    for idx in range(len(current) - 1, -1, -1):
        if time.time() - start_time > time_budget:
            break
        
        # Try removing comparator at idx
        candidate = current[:idx] + current[idx+1:]
        
        # Fast check with sample
        if _is_sorted_network_sample(candidate):
            # Full verification
            if _is_sorted_network_full(candidate):
                current = candidate
    
    return current

def _add_comparator_if_needed(network, time_budget):
    """Add comparators to make network valid if it's not already."""
    if _is_sorted_network_full(network):
        return network
    
    start_time = time.time()
    current = network[:]
    
    # Try adding comparators to fix the network
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for pair in all_pairs:
        if time.time() - start_time > time_budget:
            break
        
        candidate = current + [pair]
        if _is_sorted_network_sample(candidate):
            if _is_sorted_network_full(candidate):
                current = candidate
                break
    
    return current

def construct():
    """Construct a sorting network using greedy local search."""
    start_time = time.time()
    time_budget = DEADLINE - (time.time() - start_time) - 5
    
    if time_budget < 10:
        # Not enough time, return seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Start with odd-even merge sort (109 comparators for 17 wires)
    initial = _odd_even_merge_sort_network(N)
    
    # Ensure it's valid
    if not _is_sorted_network_full(initial):
        initial = _add_comparator_if_needed(initial, time_budget * 0.1)
    
    # Greedy local search
    best = _greedy_local_search(initial, time_budget * 0.8)
    
    # Final verification
    if not _is_sorted_network_full(best):
        # Fallback to seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
# EVOLVE-BLOCK-END

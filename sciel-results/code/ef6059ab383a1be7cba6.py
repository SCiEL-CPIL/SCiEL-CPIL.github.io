"""Construct a size-efficient sorting network on 18 wires."""
import time
import numpy as np

N = 18
DEADLINE = 82.0  # safety margin before 90s kill


def _odd_even_transposition_network(n):
    """Generate odd-even transposition sorting network (seed, 153 comparators for n=18)."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps


def _batcher_odd_merge_sort(n):
    """Batcher's odd-even merge sort network."""
    def merge(a, b, offset_a, offset_b):
        """Merge two sorted halves of length a and b."""
        comps = []
        if a == 0:
            return comps
        if b == 0:
            return comps
        if a == 1 and b == 1:
            return [(offset_a, offset_b)]
        
        if a > b:
            # Recursively merge
            half = b // 2
            comps += merge(a, half, offset_a, offset_b)
            comps += merge(a, b - half, offset_a, offset_b + half)
            # Interleaving comparators
            for i in range(min(a, b - half)):
                comps.append((offset_a + i, offset_b + half + i))
            # Final merge of remaining
            comps += merge(a - half, b - half, offset_a + half, offset_b + half)
            return comps
        else:
            half = a // 2
            comps += merge(half, b, offset_a, offset_b)
            comps += merge(a - half, b, offset_a + half, offset_b)
            for i in range(min(a - half, b)):
                comps.append((offset_a + half + i, offset_b + i))
            comps += merge(half, b, offset_a, offset_b + b // 2)
            return comps
    
    def sort(k, offset):
        """Sort k elements starting at offset."""
        if k <= 1:
            return []
        half = k // 2
        comps = []
        comps += sort(half, offset)
        comps += sort(k - half, offset + half)
        comps += merge(half, k - half, offset, offset + half)
        return comps
    
    return sort(n, 0)


def _verify_network(network, n):
    """Verify that the network sorts all 2^n Boolean inputs."""
    all_inputs = np.zeros((1 << n, n), dtype=np.uint8)
    for i in range(n):
        all_inputs[:, i] = (np.arange(1 << n) >> i) & 1
    
    state = all_inputs.copy()
    for (i, j) in network:
        state[:, i] = np.minimum(state[:, i], state[:, j])
        state[:, j] = np.maximum(state[:, i], state[:, j])
    
    for i in range(n - 1):
        if np.any((state[:, i] == 1) & (state[:, i + 1] == 0)):
            return False
    return True


def _apply_comparator(state, i, j):
    """Apply compare-exchange on wires i and j to the state array."""
    state[:, i] = np.minimum(state[:, i], state[:, j])
    state[:, j] = np.maximum(state[:, i], state[:, j])
    return state


def _count_inversions(state, n):
    """Count total inversions across all inputs (0/1 pairs where 1 appears before 0)."""
    total = 0
    for i in range(n - 1):
        total += np.sum((state[:, i] == 1) & (state[:, i + 1] == 0))
    return total


def _remove_redundant_comparators(network, n, deadline):
    """Remove comparators that don't affect correctness (greedy backward removal)."""
    if not network:
        return network
    
    # Verify initial
    if not _verify_network(network, n):
        return network
    
    # Try removing each comparator from the end (most recently added first)
    # Use a more efficient approach: try to remove from any position
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Temporarily remove comparator at idx
            test_network = network[:idx] + network[idx + 1:]
            if _verify_network(test_network, n):
                network = test_network
                improved = True
                break  # Restart scan from the beginning after removal
    
    return network


def _local_search_swap(network, n, deadline):
    """Try to improve the network by swapping or inserting comparators."""
    if not network:
        return network
    
    best_network = list(network)
    best_len = len(best_network)
    
    # Try removing one comparator and adding a different one
    for idx in range(len(best_network)):
        if time.time() > deadline:
            break
        # Remove comparator at idx
        test = best_network[:idx] + best_network[idx + 1:]
        if not _verify_network(test, n):
            continue
        best_network = test
        best_len -= 1
        # After successful removal, try to remove another
        break
    
    return best_network


def _greedy_add_comparators(start_network, n, deadline, max_comparators=4096):
    """
    Start from a valid network and try to reduce its size by:
    1. Removing redundant comparators
    2. Trying alternative comparator placements
    """
    network = list(start_network)
    
    # First pass: remove redundant comparators
    network = _remove_redundant_comparators(network, n, deadline)
    
    # Second pass: try to find shorter valid networks by replacing segments
    # For each pair of positions, try inserting a different comparator
    if time.time() < deadline - 10:
        for _ in range(3):  # Multiple passes
            if time.time() > deadline:
                break
            removed = False
            for idx in range(len(network)):
                if time.time() > deadline:
                    break
                test = network[:idx] + network[idx + 1:]
                if _verify_network(test, n):
                    network = test
                    removed = True
                    break
            if not removed:
                break
    
    return network


def construct():
    start_time = time.time()
    
    # Generate all 2^18 Boolean inputs
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1
    
    # Strategy: Start with Batcher's odd-even merge sort, then optimize
    # Batcher's OMS for 18 wires should be much better than 153
    
    # First, get Batcher's odd-even merge sort network
    batcher_network = _batcher_odd_merge_sort(N)
    
    # Verify it works
    if not _verify_network(batcher_network, N):
        # Fallback to odd-even transposition
        batcher_network = _odd_even_transposition_network(N)
    
    # Now try to optimize by removing redundant comparators
    optimized = _remove_redundant_comparators(batcher_network, N, DEADLINE)
    
    # If we still have time, try more aggressive optimization
    if time.time() < DEADLINE - 5:
        # Try greedy approach: start from a smaller network and build up
        # Or try to find a better initial network
        
        # Alternative: Use a known smaller network structure
        # For 18 wires, try a combination of stages
        
        # Let's try a different approach: start with the optimized network
        # and try local modifications
        
        # Try removing more comparators with a different order
        for pass_num in range(5):
            if time.time() > DEADLINE - 3:
                break
            removed = False
            for idx in range(len(optimized) - 1, -1, -1):
                if time.time() > DEADLINE - 3:
                    break
                test = optimized[:idx] + optimized[idx + 1:]
                if _verify_network(test, N):
                    optimized = test
                    removed = True
                    break
            if not removed:
                break
    
    # Final verification
    if not _verify_network(optimized, N):
        # Return the seed network as fallback
        return _odd_even_transposition_network(N)
    
    return optimized

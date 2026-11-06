"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # odd-even transposition for 17 wires

# EVOLVE-BLOCK-START

def _apply_network(comparators, inputs):
    """Apply network to a list of inputs; return sorted list."""
    a = list(inputs)
    for i, j in comparators:
        if a[i] > a[j]:
            a[i], a[j] = a[j], a[i]
    return a

def _verify_zero_one(comparators):
    """Verify correctness over all 2^N zero-one inputs using vectorized numpy."""
    n = len(comparators)
    if n == 0:
        return True
    # Generate all 2^N boolean vectors
    all_inputs = np.arange(2**N, dtype=np.int32)
    # Shape: (2^N, N)
    bits = np.zeros((2**N, N), dtype=np.int32)
    for w in range(N):
        bits[:, w] = (all_inputs >> w) & 1
    
    # Apply each comparator
    for i, j in comparators:
        # min to i, max to j
        bits[:, i] = np.minimum(bits[:, i], bits[:, j])
        bits[:, j] = np.maximum(bits[:, i], bits[:, j])
    
    # Check if each row is sorted (non-decreasing)
    # A row is sorted if all adjacent pairs are non-decreasing
    for w in range(N - 1):
        if np.any(bits[:, w] > bits[:, w + 1]):
            return False
    return True

def _count_comparators(comparators):
    return len(comparators)

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _random_comparator():
    i, j = random.sample(range(N), 2)
    if i > j:
        i, j = j, i
    return (i, j)

def _local_search(start_comparators, max_iterations=5000, time_limit=85.0):
    """Local search to minimize comparator count."""
    best = list(start_comparators)
    best_count = len(best)
    start_time = time.time()
    
    # Try to remove comparators
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if best_count <= 63:  # lower bound
            break
        idx = random.randint(0, len(best) - 1)
        candidate = best[:idx] + best[idx+1:]
        if _verify_zero_one(candidate):
            best = candidate
            best_count = len(best)
    
    # Try to swap positions (reorder) to enable further removals
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if best_count <= 63:
            break
        if len(best) < 2:
            break
        i, j = random.sample(range(len(best)), 2)
        candidate = list(best)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        # Reordering doesn't change count, but might enable removal
        # Just check if it's still valid (it should be, since same set)
        # Actually reordering the same comparators keeps validity if the set is the same
        # So this is useless unless we also try removals after reordering
        # Skip pure reordering for now
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    best_network = _seed_network()
    best_count = len(best_network)
    
    # Phase 1: Local search from seed
    if time.time() < deadline:
        candidate = _local_search(best_network, max_iterations=10000, time_limit=deadline - time.time())
        if len(candidate) < best_count:
            best_network = candidate
            best_count = len(candidate)
    
    # Phase 2: Try different random starts
    while time.time() < deadline - 5.0:
        # Start from a random network and try to build up? No, that's hard.
        # Instead, try more aggressive local search with different seeds
        random.seed(int(time.time() * 1000) % (2**31))
        candidate = _local_search(best_network, max_iterations=5000, time_limit=deadline - time.time())
        if len(candidate) < best_count:
            best_network = candidate
            best_count = len(candidate)
    
    return best_network

# EVOLVE-BLOCK-END

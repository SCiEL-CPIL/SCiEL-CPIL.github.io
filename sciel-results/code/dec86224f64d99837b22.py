"""Improved program for sorting networks on 17 wires with efficient validation and search."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Pre-compute the input matrix globally to avoid repeated allocation
# All 2^17 binary inputs as a (131072, 17) boolean array
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_GLOBAL_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_GLOBAL_SORTED = np.sort(_GLOBAL_INPUTS, axis=1)

def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses pre-computed global inputs for speed.
    """
    if len(comparators) > 4096:
        return False
    
    # Simulate the network on all inputs
    state = _GLOBAL_INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    # Check if state matches sorted_inputs
    if not np.array_equal(state, _GLOBAL_SORTED):
        return False
    
    return True

def count_comparators(comparators):
    """Count the number of comparators."""
    return len(comparators)

def odd_even_transposition_network():
    """Generate the standard odd-even transposition network for N wires."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def try_remove_comparators(comparators, max_attempts=50, deadline=None):
    """
    Try to remove redundant comparators from the network.
    Returns a potentially smaller valid network.
    """
    current = list(comparators)
    
    attempts = 0
    while True:
        if deadline and time.time() > deadline:
            break
        if not current:
            break
        if attempts >= max_attempts:
            break
        
        attempts += 1
        idx = random.randint(0, len(current) - 1)
        
        # Try removing the comparator at idx
        candidate = current[:idx] + current[idx+1:]
        if is_sorted_network(candidate):
            current = candidate
        else:
            # Try a few more random positions
            found = False
            for _ in range(3):
                idx2 = random.randint(0, len(current) - 1)
                candidate = current[:idx2] + current[idx2+1:]
                if is_sorted_network(candidate):
                    current = candidate
                    found = True
                    break
            if not found:
                break
    
    return current

def local_search(comparators, max_iterations=200, deadline=None):
    """
    Local search: try removing, replacing, or moving comparators.
    """
    current = list(comparators)
    best = current
    best_len = len(best)
    
    for _ in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        if not current:
            break
        
        idx = random.randint(0, len(current) - 1)
        improved = False
        
        # Strategy 1: Remove
        candidate = current[:idx] + current[idx+1:]
        if is_sorted_network(candidate):
            current = candidate
            if len(current) < best_len:
                best = current
                best_len = len(best)
            improved = True
        
        # Strategy 2: Replace with a different comparator
        if not improved:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = current.copy()
            candidate[idx] = (i, j)
            if is_sorted_network(candidate):
                current = candidate
                if len(current) < best_len:
                    best = current
                    best_len = len(best)
                improved = True
        
        # Strategy 3: Move to a different position
        if not improved and idx > 0:
            idx2 = random.randint(0, idx - 1)
            candidate = current.copy()
            c = candidate.pop(idx)
            candidate.insert(idx2, c)
            if is_sorted_network(candidate):
                current = candidate
                if len(current) < best_len:
                    best = current
                    best_len = len(best)
                improved = True
    
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin: finish before 90s
    
    # Start with the odd-even transposition network
    best_network = odd_even_transposition_network()
    best_len = len(best_network)
    
    # Verify seed is correct
    if not is_sorted_network(best_network):
        # This should never happen, but handle gracefully
        return best_network
    
    # Phase 1: Aggressive deletion from the seed network
    # Try to remove as many comparators as possible
    current = try_remove_comparators(best_network, max_attempts=200, deadline=deadline - 30.0)
    if len(current) < best_len:
        best_network = current
        best_len = len(best_network)
    
    # Phase 2: Local search to find further improvements
    if time.time() < deadline - 10.0:
        improved = local_search(best_network, max_iterations=500, deadline=deadline - 5.0)
        if len(improved) < best_len:
            best_network = improved
            best_len = len(improved)
    
    # Phase 3: Try random restarts with smaller initial networks
    # Generate random networks and try to improve them
    if time.time() < deadline - 5.0:
        for target_size in range(best_len - 1, max(best_len - 10, 60), -1):
            if time.time() > deadline - 5.0:
                break
            # Try a few random networks of this size
            for _ in range(3):
                if time.time() > deadline - 5.0:
                    break
                rand_net = random_network(target_size)
                if is_sorted_network(rand_net):
                    best_network = rand_net
                    best_len = target_size
                    break
    
    # Final validation
    if not is_sorted_network(best_network):
        # Return the seed as fallback
        return odd_even_transposition_network()
    
    return best_network
# EVOLVE-BLOCK-END

"""Size-efficient sorting network on 17 wires via greedy removal search."""
import time
import numpy as np

N = 17
DEADLINE = 85.0  # safety margin before 90s kill

def _all_boolean_inputs():
    """Return array of shape (2^N, N) with all Boolean inputs, uint8."""
    n = 1 << N
    # Each row is a distinct 17-bit pattern
    # Use bit extraction: input k has bit i set if (k >> i) & 1
    # Build as (N, n) then transpose for cache, but we simulate column-wise
    # Actually shape (n, N) is fine for row-wise simulation
    idx = np.arange(n, dtype=np.uint32)
    arr = np.zeros((n, N), dtype=np.uint8)
    for i in range(N):
        arr[:, i] = (idx >> i) & 1
    return arr

def _simulate(network, inputs):
    """Simulate network on all inputs. inputs shape (n, N) uint8. Returns sorted array."""
    state = inputs.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        # min goes to i, max goes to j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return state

def _is_sorted(state):
    """Check if state (n, N) is sorted in each row (non-decreasing)."""
    # Check adjacent pairs
    return np.all(state[:, :-1] <= state[:, 1:])

def _is_valid_network(network):
    """Check network sorts all Boolean inputs via zero-one principle."""
    inputs = _all_boolean_inputs()
    state = _simulate(network, inputs)
    return _is_sorted(state)

def _build_seed_network():
    """Build a seed sorting network for 17 wires.
    Use odd-even transposition as a guaranteed-correct seed (136 comparators).
    """
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _try_remove_comparator(network, index, all_inputs):
    """Try removing comparator at index; return True if still valid."""
    reduced = network[:index] + network[index+1:]
    state = _simulate(reduced, all_inputs)
    return _is_sorted(state)

def _greedy_remove(network, all_inputs, start_time, time_budget):
    """Iteratively remove redundant comparators."""
    best = list(network)
    best_count = len(best)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        improved = False
        # Try removing from the end first (often less critical)
        for idx in range(len(best) - 1, -1, -1):
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                break
            if _try_remove_comparator(best, idx, all_inputs):
                best = best[:idx] + best[idx+1:]
                best_count = len(best)
                improved = True
                break  # restart scan after removal
        if not improved:
            break
    return best

def _depth_of_network(network):
    """Compute parallel depth (longest path)."""
    # Wire depth tracking
    wire_depth = [0] * N
    for (i, j) in network:
        d = max(wire_depth[i], wire_depth[j]) + 1
        wire_depth[i] = d
        wire_depth[j] = d
    return max(wire_depth) if wire_depth else 0

def construct():
    start_time = time.time()
    time_budget = DEADLINE - start_time
    
    # Build seed
    network = _build_seed_network()
    
    # Precompute all Boolean inputs
    all_inputs = _all_boolean_inputs()
    
    # Verify seed is valid
    if not _is_valid_network(network):
        # Shouldn't happen, but return seed anyway
        return network
    
    # Greedy removal
    network = _greedy_remove(network, all_inputs, start_time, time_budget)
    
    # Final validation
    if not _is_valid_network(network):
        return _build_seed_network()
    
    return network

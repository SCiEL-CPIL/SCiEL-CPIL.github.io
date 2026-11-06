"""Size-efficient sorting networks on 17 wires."""
import time
import numpy as np

N = 17

def _all_boolean_inputs(n):
    """Generate all 2^n boolean inputs as a 2D array of shape (2^n, n)."""
    total = 1 << n
    indices = np.arange(total, dtype=np.int32)[:, None]
    bit_positions = np.arange(n, dtype=np.int32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.int8)

def _simulate_network(network, inputs):
    """Simulate a compare-exchange network on all inputs."""
    state = inputs.copy()
    for i, j in network:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _is_sorted_fast(state):
    """Check if all rows are sorted (non-decreasing) using vectorized ops."""
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)

def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)

def _try_remove_comparator(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    return _is_sorting_network_fast(new_network, all_inputs)

def _try_remove_multiple(network, indices_to_remove, all_inputs):
    """Try removing multiple comparators at once. Returns True if still sorts."""
    indices_set = set(indices_to_remove)
    new_network = [c for i, c in enumerate(network) if i not in indices_set]
    return _is_sorting_network_fast(new_network, all_inputs)

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators with multiple strategies."""
    network = list(network)
    
    # Strategy 1: Batch removal from the end - try removing 3 at a time
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return network
        # Try removing batches of 3 from the end
        for start in range(len(network) - 3, -1, -3):
            if time.time() > deadline:
                return network
            batch = [start, start + 1, start + 2]
            if _try_remove_multiple(network, batch, all_inputs):
                indices_set = set(batch)
                network = [c for i, c in enumerate(network) if i not in indices_set]
                improved = True
                break
    
    # Strategy 2: Individual removal, prioritizing later comparators
    improved = True
    while improved:
        improved = False
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                break
    
    # Strategy 3: Try from the beginning
    improved = True
    while improved:
        improved = False
        for i in range(len(network)):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                break
    
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    # Local search: try to remove comparators
    network = _local_search_remove(network, all_inputs, deadline)
    
    return network

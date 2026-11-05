"""Improved sorting network search for 17 wires."""
N = 17
import time
import numpy as np

def _all_boolean_inputs(n):
    """Generate all 2^n boolean inputs as a 2D array of shape (2^n, n)."""
    total = 1 << n
    indices = np.arange(total, dtype=np.int32)[:, None]
    bit_positions = np.arange(n, dtype=np.int32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.int8)

def _simulate_network(network, inputs):
    """Simulate a compare-exchange network on all inputs using integer arithmetic.
    state: shape (num_inputs, n)
    network: list of (i, j) comparators
    """
    state = inputs.copy()
    for i, j in network:
        a = state[:, i].astype(np.int32)
        b = state[:, j].astype(np.int32)
        # min = (a+b - |a-b|)/2, max = (a+b + |a-b|)/2
        s = a + b
        d = np.abs(a - b)
        state[:, i] = ((s - d) // 2).astype(np.int8)
        state[:, j] = ((s + d) // 2).astype(np.int8)
    return state

def _is_sorted_fast(final):
    """Check if all rows are sorted (non-decreasing). Returns True if all sorted."""
    diffs = np.diff(final.astype(np.int8), axis=1)
    return not np.any(diffs < 0)

def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)

def _try_remove_comparator(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    return _is_sorting_network_fast(new_network, all_inputs)

def _batch_try_removals(network, all_inputs, candidates):
    """Try removing multiple comparators. Returns list of indices that can be removed.
    Uses a faster approach: simulate with each candidate removed and check sorting.
    """
    removable = []
    for idx in candidates:
        new_network = network[:idx] + network[idx+1:]
        if _is_sorting_network_fast(new_network, all_inputs):
            removable.append(idx)
    return removable

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators with smarter ordering."""
    network = list(network)
    improved = True
    while improved:
        improved = False
        # Try removals in reverse order first (later comparators may be more redundant)
        # Also try in forward order
        for direction in [range(len(network) - 1, -1, -1), range(len(network))]:
            for i in direction:
                if time.time() > deadline:
                    return network
                if _try_remove_comparator(network, i, all_inputs):
                    network.pop(i)
                    improved = True
                    break
            if improved:
                break
    return network

def _local_search_remove_batched(network, all_inputs, deadline):
    """Local search that tests multiple removals per pass to find all removable at once."""
    network = list(network)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return network
        # Find all removable comparators in one pass
        removable = _batch_try_removals(network, all_inputs, range(len(network)))
        if removable:
            # Remove them in reverse order to preserve indices
            for idx in sorted(removable, reverse=True):
                network.pop(idx)
            improved = True
    return network

def construct():
    start_time = time.time()
    # 80s deadline to be safe within 90s limit
    deadline = start_time + 80.0
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # First, verify the initial network is correct
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    # Local search: try to remove comparators
    network = _local_search_remove(network, all_inputs, deadline)
    
    # Try batched removal as well
    if time.time() < deadline:
        network = _local_search_remove_batched(network, all_inputs, deadline)
    
    return network
# EVOLVE-BLOCK-END

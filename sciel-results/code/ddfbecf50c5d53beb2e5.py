"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17
import time
import numpy as np

def _all_boolean_inputs(n):
    """Generate all 2^n boolean inputs as a 2D array of shape (2^n, n)."""
    total = 1 << n
    # Use bit extraction: input k has bit i = (k >> i) & 1
    indices = np.arange(total, dtype=np.int32)[:, None]
    bit_positions = np.arange(n, dtype=np.int32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.int8)

def _simulate_network(network, inputs):
    """Simulate a compare-exchange network on all inputs.
    inputs: shape (num_inputs, n)
    network: list of (i, j) comparators
    Returns the final state after all comparators.
    """
    state = inputs.copy()
    for i, j in network:
        # Compute min and max from current state before assignment to avoid view mutation
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _is_sorting_network(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    # Check if each row is sorted: all 0s followed by all 1s
    # A row is sorted if it's non-decreasing
    for row in final:
        for k in range(len(row) - 1):
            if row[k] > row[k + 1]:
                return False
    return True

def _is_sorting_network_fast(network, all_inputs):
    """Faster check: verify each row is of form 0...01...1 using vectorized ops"""
    final = _simulate_network(network, all_inputs)
    # A row is sorted (non-decreasing) if there are no instances of 1 followed by 0.
    # i.e., diff[j] < 0 implies final[j] > final[j+1], which is bad for non-decreasing.
    # For boolean 0/1, sorted means 0...0 1...1.
    # np.diff(final, axis=1) gives differences.
    # If any diff is -1 (1->0), it's not sorted.
    diffs = np.diff(final, axis=1)
    # Check if any row has a negative difference
    invalid_rows = np.any(diffs < 0, axis=1)
    return not np.any(invalid_rows)

def _try_remove_comparator(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    return _is_sorting_network_fast(new_network, all_inputs)

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators."""
    network = list(network)
    improved = True
    while improved:
        improved = False
        for i in range(len(network)):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                break  # Restart after a removal
    return network

def construct():
    start_time = time.time()
    # 50s deadline to be safe within 60s limit
    deadline = start_time + 50.0
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # First, verify the initial network is correct
    if not _is_sorting_network_fast(network, all_inputs):
        # Should not happen, but return as-is
        return network
    
    # Local search: try to remove comparators
    network = _local_search_remove(network, all_inputs, deadline)
    
    return network
# EVOLVE-BLOCK-END

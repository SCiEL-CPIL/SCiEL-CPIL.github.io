"""Size-efficient sorting networks on 17 wires - optimized local search."""
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
    """Simulate a compare-exchange network on all inputs."""
    state = inputs.copy()
    for i, j in network:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _is_sorted_fast(state):
    """Check if all rows are sorted (non-decreasing)."""
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)

def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)

def _count_unsorted(final):
    """Count how many inputs are unsorted."""
    diffs = np.diff(final, axis=1)
    return np.sum(diffs < 0)

def _try_remove_comparator(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    final = _simulate_network(new_network, all_inputs)
    return _is_sorted_fast(final)

def _find_removable_batch(network, all_inputs, start_idx=0, end_idx=None):
    """Find all removable comparators in a range using batch verification.
    Returns list of indices that can be removed."""
    if end_idx is None:
        end_idx = len(network)
    
    removable = []
    for idx in range(end_idx - 1, start_idx - 1, -1):
        new_network = network[:idx] + network[idx+1:]
        final = _simulate_network(new_network, all_inputs)
        if _is_sorted_fast(final):
            removable.append(idx)
    return removable

def _local_search_aggressive(network, all_inputs, deadline):
    """Aggressive local search: repeatedly try to remove comparators."""
    network = list(network)
    
    # Phase 1: Single removal passes from both ends
    for _pass in range(10):
        if time.time() > deadline:
            break
        improved = False
        
        # Try from the end
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            new_network = network[:i] + network[i+1:]
            final = _simulate_network(new_network, all_inputs)
            if _is_sorted_fast(final):
                network = new_network
                improved = True
                break
        
        if not improved:
            # Try from the beginning
            for i in range(len(network)):
                if time.time() > deadline:
                    return network
                new_network = network[:i] + network[i+1:]
                final = _simulate_network(new_network, all_inputs)
                if _is_sorted_fast(final):
                    network = new_network
                    improved = True
                    break
        
        if not improved:
            break
    
    return network

def _try_swap_and_remove(network, i, all_inputs):
    """Try swapping network[i] and network[i+1], then removing one of them."""
    if i >= len(network) - 1:
        return None
    if network[i] == network[i+1]:
        return None
    
    new_network = network.copy()
    new_network[i], new_network[i+1] = new_network[i+1], new_network[i]
    
    # Try removing i
    test1 = new_network[:i] + new_network[i+1:]
    final1 = _simulate_network(test1, all_inputs)
    if _is_sorted_fast(final1):
        return test1
    
    # Try removing i+1
    test2 = new_network[:i+1] + new_network[i+2:]
    final2 = _simulate_network(test2, all_inputs)
    if _is_sorted_fast(final2):
        return test2
    
    return None

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
    
    # Phase 1: Aggressive single-removal local search
    network = _local_search_aggressive(network, all_inputs, deadline)
    
    # Phase 2: Swap-and-remove to enable further reductions
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(network) - 1):
                if time.time() > deadline - 5.0:
                    break
                result = _try_swap_and_remove(network, i, all_inputs)
                if result is not None:
                    network = result
                    improved = True
                    break
    
    # Phase 3: Another round of single-removal search after swaps
    if time.time() < deadline - 2.0:
        network = _local_search_aggressive(network, all_inputs, deadline)
    
    return network

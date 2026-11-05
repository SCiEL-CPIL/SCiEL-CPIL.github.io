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

def _simulate_network_skip(network, inputs, skip_idx):
    """Simulate a compare-exchange network, skipping one comparator if needed.
    state: shape (num_inputs, n)
    network: list of (i, j) comparators
    skip_idx: index of comparator to skip, or -1 for none
    """
    state = inputs.copy()
    for idx, (i, j) in enumerate(network):
        if idx == skip_idx:
            continue
        a = state[:, i]
        b = state[:, j]
        state[:, i] = a & b
        state[:, j] = a | b
    return state

def _is_sorted_fast(final):
    """Check if all rows are sorted (non-decreasing). Returns True if all sorted."""
    return not np.any(final[:, :-1] > final[:, 1:])

def _is_sorting_network_skip_fast(network, all_inputs, skip_idx):
    """Check if network sorts all boolean inputs, skipping one comparator."""
    final = _simulate_network_skip(network, all_inputs, skip_idx)
    return _is_sorted_fast(final)

def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network_skip(network, all_inputs, -1)
    return _is_sorted_fast(final)

def construct():
    start_time = time.time()
    # Deadline set to 85s to allow 5s safety margin
    deadline = start_time + 85.0
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network
        
    # Local search: Greedy removal
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(network):
            if time.time() > deadline:
                return network
            
            # Try removing comparator at i
            if _is_sorting_network_skip_fast(network, all_inputs, i):
                # Remove it
                network.pop(i)
                improved = True
                # Do not increment i
            else:
                i += 1
                
    return network

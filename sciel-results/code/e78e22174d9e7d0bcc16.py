import time
import random
import numpy as np

N = 17
TOTAL_INPUTS = 1 << N  # 131072

def _generate_test_cases():
    """
    Generate all 2^N boolean inputs as a numpy array of shape (TOTAL_INPUTS, N).
    Input m is the binary representation of m.
    """
    # Create an array of indices 0 to 2^N - 1
    indices = np.arange(TOTAL_INPUTS, dtype=np.int32)
    # Extract bits for each wire
    # Wire k corresponds to bit k
    # We can do this by shifting and masking
    # However, creating a 131072 x 17 array of ints is ~2MB, which is fine.
    # We want row m to be the binary representation of m.
    # bit k of m is (m >> k) & 1
    
    # Efficient way: 
    # Create a matrix where column k is (indices >> k) & 1
    cols = []
    for k in range(N):
        cols.append((indices >> k) & 1)
    # Stack columns to form rows
    # np.column_stack creates a 2D array where each row is the inputs
    inputs = np.column_stack(cols).T  # Transpose to get (N, TOTAL_INPUTS) for easier column ops? 
    # Actually, let's keep it (TOTAL_INPUTS, N) so we can operate on rows?
    # Comparators operate on two wires (columns) for all inputs (rows).
    # So shape (TOTAL_INPUTS, N) is good.
    # Let's re-verify:
    # indices: (TOTAL_INPUTS,)
    # (indices >> k) & 1: (TOTAL_INPUTS,)
    # column_stack(cols) -> (TOTAL_INPUTS, N)
    # This is correct.
    return np.column_stack(cols)

def _simulate_network_vectorized(network, inputs):
    """
    Simulates the network on the given inputs matrix.
    inputs: numpy array of shape (TOTAL_INPUTS, N)
    network: list of (i, j) tuples
    Returns: sorted numpy array of shape (TOTAL_INPUTS, N)
    """
    # Make a copy to avoid modifying the original
    state = inputs.copy()
    for i, j in network:
        # Extract columns i and j
        a = state[:, i]
        b = state[:, j]
        # min goes to i, max goes to j
        # min(a, b)
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return state

def _is_valid_network_fast(network, inputs):
    """
    Checks if the network sorts all inputs.
    """
    final_state = _simulate_network_vectorized(network, inputs)
    # Check if columns are sorted: col k <= col k+1 for all k
    # i.e., final_state[:, k] <= final_state[:, k+1]
    # We can check if there is any violation.
    # Violation if final_state[:, k] > final_state[:, k+1]
    # Sum of violations should be 0.
    # (final_state[:, :-1] > final_state[:, 1:]).any()
    # This creates a boolean array of shape (TOTAL_INPUTS, N-1)
    # Using .any() is efficient.
    return not (final_state[:, :-1] > final_state[:, 1:]).any()

def _greedy_remove_fast(network, inputs, deadline):
    """
    Greedily removes redundant comparators using fast vectorized verification.
    """
    current_net = list(network)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate backwards to remove items
        for k in range(len(current_net) - 1, -1, -1):
            if time.time() >= deadline:
                break
            # Try removing comparator at position k
            # Optimization: Skip if this comparator is "essential" based on simple heuristics?
            # For now, just try removal.
            candidate_net = current_net[:k] + current_net[k+1:]
            if _is_valid_network_fast(candidate_net, inputs):
                current_net = candidate_net
                changed = True
                # Reset loop to start from the end again? 
                # Or just break and let the outer while restart?
                # Breaking inner loop and restarting outer while is safer.
                break
    return current_net

def _local_search_improve_fast(network, inputs, deadline):
    """
    Local search to improve the network.
    """
    best_net = list(network)
    best_len = len(best_net)
    
    # Limit iterations based on time
    start = time.time()
    while time.time() < deadline:
        current_net = list(best_net)
        move_type = random.randint(0, 2)
        
        if move_type == 0:
            # Insert a new comparator
            pos = random.randint(0, len(current_net))
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            current_net.insert(pos, (i, j))
        elif move_type == 1:
            # Remove a comparator
            if len(current_net) > 1:
                pos = random.randint(0, len(current_net) - 1)
                current_net.pop(pos)
        else:
            # Swap two comparators
            if len(current_net) >= 2:
                k1 = random.randint(0, len(current_net) - 1)
                k2 = random.randint(0, len(current_net) - 1)
                if k1 != k2:
                    current_net[k1], current_net[k2] = current_net[k2], current_net[k1]
        
        if not _is_valid_network_fast(current_net, inputs):
            continue
        
        # Try to remove redundant comparators (limited scope to save time)
        # Do a few rounds of greedy removal
        for _ in range(3):
             if time.time() >= deadline: break
             current_net = _greedy_remove_fast(current_net, inputs, deadline)
        
        if len(current_net) < best_len:
            best_net = current_net
            best_len = len(current_net)
            
    return best_net

def construct():
    start_time = time.time()
    # Safety margin: 60s limit. Let's aim to finish by 45s.
    deadline = start_time + 45.0
    
    # 1. Generate initial network: Odd-Even Transposition
    # This is a known valid sorting network.
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    # 2. Pre-compute test cases (all 2^N inputs)
    # This takes a bit of time/memory but is done once.
    inputs = _generate_test_cases()
    
    # 3. Verify initial network (should be valid)
    # We assume it is, but let's check just in case.
    if not _is_valid_network_fast(network, inputs):
        # Fallback if something went wrong (should not happen)
        return network
    
    # 4. Greedy removal phase
    # Allocate 60% of time to this
    greedy_deadline = start_time + (deadline - start_time) * 0.6
    network = _greedy_remove_fast(network, inputs, greedy_deadline)
    
    # 5. Local search phase
    # Allocate remaining time to this
    network = _local_search_improve_fast(network, inputs, deadline)
    
    # 6. Final greedy removal
    network = _greedy_remove_fast(network, inputs, deadline)
    
    # 7. Final verification
    if _is_valid_network_fast(network, inputs):
        return network
    else:
        # Fallback to a known valid network
        fallback = []
        for phase in range(N):
            start = phase % 2
            for i in range(start, N - 1, 2):
                fallback.append((i, i + 1))
        return fallback

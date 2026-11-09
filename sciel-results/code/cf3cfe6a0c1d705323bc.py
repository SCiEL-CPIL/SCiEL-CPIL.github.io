import time
import numpy as np

N = 18
_SEED_COMPARATORS = 153

def _apply_comparator(state, i, j):
    """Apply a compare-exchange operation to a state array."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _is_sorted_network_vectorized(comparators):
    """Verify that a network of comparators sorts all Boolean inputs using NumPy."""
    # Generate all 2^N Boolean inputs
    # 2^18 = 262144
    # Creating this array takes memory and time, but it's done once.
    # We can cache it? No, state changes.
    # But the inputs are constant.
    
    # Efficient generation of all binary vectors
    # Use np.arange(2**N) and bitwise operations
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1
        
    state = inputs.copy()
    
    for i, j in comparators:
        # Compare and exchange
        # min goes to i, max goes to j
        # If state[i] > state[j], swap
        # min_val = np.minimum(state[:, i], state[:, j])
        # max_val = np.maximum(state[:, i], state[:, j])
        # state[:, i] = min_val
        # state[:, j] = max_val
        
        # This is faster than swap logic
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check if sorted
    # A vector is sorted if state[:, k] <= state[:, k+1] for all k
    # Or simply check if state equals its sorted version
    # np.sort is expensive.
    # Check diffs
    diffs = np.diff(state, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _count_comparators(comparators):
    """Count the number of comparators in a network."""
    return len(comparators)

def _depth(comparators):
    """Calculate the parallel depth of a sorting network."""
    if not comparators:
        return 0
    
    wire_depth = np.zeros(N, dtype=int)
    depth = 0
    for i, j in comparators:
        wire_depth[i] = max(wire_depth[i], depth)
        wire_depth[j] = max(wire_depth[j], depth)
        depth = max(wire_depth[i], wire_depth[j]) + 1
    
    return int(depth)

def construct():
    """Construct a minimal sorting network using local search."""
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # Start with the seed odd-even transposition network
    best_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    best_count = len(best_network)
    
    # Pre-compute inputs for verification?
    # The vectorized verification creates the input array every time.
    # This is expensive. Let's create it once.
    # But 2^18 * 18 bytes = ~4.7 MB. This is fine.
    # However, copying it every time is also expensive.
    # We can pass the inputs to the verifier.
    
    # Let's modify _is_sorted_network_vectorized to accept inputs
    def verify(comparators, inputs):
        state = inputs.copy()
        for i, j in comparators:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        diffs = np.diff(state, axis=1)
        return not np.any(diffs < 0)
        
    # Generate inputs once
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(2**N) >> (N - 1 - i)) & 1
        
    current_network = best_network[:]
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removals
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            candidate = current_network[:idx] + current_network[idx+1:]
            if verify(candidate, inputs):
                current_network = candidate
                improved = True
                # Restart loop to try more removals
                break
        
        if not improved:
            # Try swaps
            for idx in range(len(current_network) - 1):
                if time.time() > deadline:
                    break
                
                # Swap idx and idx+1
                candidate = current_network[:]
                candidate[idx], candidate[idx+1] = current_network[idx+1], current_network[idx]
                
                if verify(candidate, inputs):
                    current_network = candidate
                    improved = True
                    break
                    
    best_network = current_network
    best_count = len(best_network)
    
    return best_network

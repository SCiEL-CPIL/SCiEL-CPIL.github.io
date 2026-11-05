import numpy as np
import time
import random

N = 16
TOTAL_INPUTS = 1 << N

def _generate_initial_state():
    """
    Generate the initial state array for all 2^N inputs.
    Shape: (N, 2^N)
    state[i, j] is the value on wire i for input j.
    Input j is an N-bit integer. Bit i of j is the value on wire i.
    """
    inputs = np.arange(TOTAL_INPUTS, dtype=np.uint32)
    # We want state[i, j] = (j >> i) & 1
    # We can do this by reshaping and broadcasting or using bitwise ops.
    # Efficient way:
    # Create a vector of powers of 2 for wires? No, just shift.
    # state = np.zeros((N, TOTAL_INPUTS), dtype=np.uint8)
    # for i in range(N):
    #     state[i] = (inputs >> i) & 1
    # This is fast enough.
    state = np.zeros((N, TOTAL_INPUTS), dtype=np.uint8)
    for i in range(N):
        state[i] = (inputs >> i) & 1
    return state

INITIAL_STATE = _generate_initial_state()

def _verify(nets):
    """
    Verify if the network sorts all inputs.
    Returns True if valid.
    """
    if not nets:
        return False
        
    # Copy the initial state
    state = INITIAL_STATE.copy()
    
    for i, j in nets:
        # Compare-exchange: min to i, max to j
        # Note: We assume wire i < wire j in the output order? 
        # The problem says: "min to wire i and max to j".
        # So if we want sorted order [min, ..., max], we should ensure 
        # that the "min" goes to the lower index if we want ascending order.
        # The check `np.diff(state, axis=0) >= 0` checks if state[0] <= state[1] <= ...
        # So we must map min -> lower index, max -> higher index.
        
        min_ij = np.minimum(state[i], state[j])
        max_ij = np.maximum(state[i], state[j])
        state[i] = min_ij
        state[j] = max_ij
        
    # Check if the result is sorted in non-decreasing order
    # diff(state, axis=0) gives state[1]-state[0], state[2]-state[1], etc.
    # If all are >= 0, it is sorted.
    diffs = np.diff(state, axis=0)
    return np.all(diffs >= 0)

def _batcher_sort(n):
    """
    Generate Batcher's Odd-Even Merge Sort network.
    Returns a list of (i, j) pairs.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort the two halves
    left = _batcher_sort(half)
    right = [(i + half, j + half) for i, j in left]
    
    # Odd-Even Merge
    merge = _odd_even_merge(n)
    
    return left + right + merge

def _odd_even_merge(n):
    """
    Generate the Odd-Even Merge network for n wires.
    Assumes n is a power of 2.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    merge = []
    
    # Phase 1: Compare (i, i + n/2) for i in 0..n/2-1
    for i in range(half):
        merge.append((i, i + half))
        
    # Phase 2: Recursively merge odd and even positions
    # The odd positions in the merged array correspond to indices 1, 3, 5...
    # The even positions correspond to 0, 2, 4...
    # We need to map the recursive merge on n/2 wires to these positions.
    
    # Recursive merge on n/2 wires
    sub_merge = _odd_even_merge(half)
    
    # Apply to odd wires: 1, 3, 5, ..., n-1
    # Index k in sub_merge corresponds to wire 2*k + 1 in original
    for i, j in sub_merge:
        merge.append((2 * i + 1, 2 * j + 1))
        
    # Apply to even wires: 0, 2, 4, ..., n-2
    # Index k in sub_merge corresponds to wire 2*k in original
    for i, j in sub_merge:
        merge.append((2 * i, 2 * j))
        
    return merge

def _odd_even_transposition(n):
    """
    Generate Odd-Even Transposition network.
    """
    nets = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            nets.append((i, i + 1))
    return nets

def _local_search(nets, deadline):
    """
    Iteratively remove comparators if the network remains valid.
    """
    current_nets = list(nets)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias towards early removal
        indices = list(range(len(current_nets)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            # Try removing current_nets[idx]
            candidate = current_nets[:idx] + current_nets[idx+1:]
            
            if _verify(candidate):
                current_nets = candidate
                improved = True
                
    return current_nets

def construct():
    start_time = time.time()
    # Deadline with safety margin
    deadline = start_time + 80.0
    
    # Generate candidate starting networks
    candidates = []
    
    # Batcher's Sort
    batcher_net = _batcher_sort(N)
    candidates.append(batcher_net)
    
    # Odd-Even Transposition
    oet_net = _odd_even_transposition(N)
    candidates.append(oet_net)
    
    best_net = None
    best_len = float('inf')
    
    for net in candidates:
        if time.time() > deadline - 5.0:
            break
            
        if not _verify(net):
            continue
            
        # Local search
        optimized_net = _local_search(net, deadline - 2.0)
        
        if len(optimized_net) < best_len:
            best_len = len(optimized_net)
            best_net = optimized_net
            
        # If we have time, try to improve further with random restarts/perturbations
        # This can help escape local minima
        if time.time() < deadline - 10.0:
            # Try a few random perturbations
            for _ in range(5):
                if time.time() > deadline - 5.0:
                    break
                
                # Create a perturbed version
                perturbed = list(optimized_net)
                # Add a random comparator
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                perturbed.append((i, j))
                
                # Try to remove comparators from this perturbed network
                # This might lead to a different local minimum
                new_optimized = _local_search(perturbed, deadline - 5.0)
                
                if len(new_optimized) < best_len:
                    best_len = len(new_optimized)
                    best_net = new_optimized
                    optimized_net = new_optimized # Update the base for further perturbations if improved

    if best_net is None:
        # Fallback to a known valid network
        return oet_net
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    # Verify one last time
    if _verify(net):
        print("Verification passed")
    else:
        print("Verification failed")

import numpy as np
import time
import random

N = 17
DEADLINE = 85.0  # Safety margin before 90s kill

# Precompute all 2^17 boolean inputs
# Shape: (131072, 17), dtype uint8
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_GLOBAL_INPUTS = ((_indices[:, None] & _bits[None, :]) > 0).astype(np.uint8)

def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses bitwise operations for speed.
    """
    if len(comparators) > 4096:
        return False
    
    state = _GLOBAL_INPUTS.copy()
    
    for i, j in comparators:
        wa = state[:, i]
        wb = state[:, j]
        # For booleans: min = AND, max = OR
        state[:, i] = wa & wb
        state[:, j] = wa | wb
        
    # Check if sorted: no adjacent pair where left > right (1 > 0)
    if np.any(state[:, :-1] > state[:, 1:]):
        return False
        
    return True

def odd_even_transposition_network():
    """Generate the standard odd-even transposition network for N wires."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with seed
    best_network = odd_even_transposition_network()
    best_len = len(best_network)
    
    # Verify seed
    if not is_sorted_network(best_network):
        return best_network

    current = best_network[:]
    
    # Phase 1: Greedy Deletion
    # Try to remove comparators one by one.
    # We do this in multiple passes.
    
    improved = True
    while improved and (time.time() < deadline - 10.0):
        improved = False
        # Shuffle to randomize order
        random.shuffle(current)
        
        # Iterate backwards to handle removals safely
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 10.0:
                break
                
            # Try to remove current[idx]
            candidate = current[:idx] + current[idx+1:]
            if is_sorted_network(candidate):
                current = candidate
                improved = True
                if len(current) < best_len:
                    best_network = current[:]
                    best_len = len(current)

    # Phase 2: Local Search
    # Try to improve by swapping, moving, or removing.
    # This helps escape local minima where no single removal works.
    
    current = best_network[:]
    last_improvement = time.time()
    
    while time.time() < deadline - 5.0:
        if time.time() - last_improvement > 5.0:
            # Stagnation: shuffle to randomize search direction
            random.shuffle(current)
            last_improvement = time.time()
            
        idx = random.randint(0, len(current) - 1)
        improved = False
        
        # 1. Try to remove
        candidate = current[:idx] + current[idx+1:]
        if is_sorted_network(candidate):
            current = candidate
            if len(current) < best_len:
                best_network = current[:]
                best_len = len(current)
            last_improvement = time.time()
            improved = True
            continue # Restart loop with new length
            
        # 2. Try to swap with random other
        if not improved:
            idx2 = random.randint(0, len(current) - 1)
            if idx2 != idx:
                candidate = current[:]
                candidate[idx], candidate[idx2] = candidate[idx2], candidate[idx]
                if is_sorted_network(candidate):
                    current = candidate
                    last_improvement = time.time()
                    improved = True
                    continue
        
        # 3. Try to move to random position
        if not improved:
            idx2 = random.randint(0, len(current) - 1)
            if idx2 != idx:
                candidate = current[:]
                comp = candidate.pop(idx)
                # Adjust idx2 if popping shifted it
                if idx < idx2:
                    idx2 -= 1
                candidate.insert(idx2, comp)
                if is_sorted_network(candidate):
                    current = candidate
                    last_improvement = time.time()
                    improved = True
                    continue

    # Final check
    if is_sorted_network(best_network):
        return best_network
    else:
        return odd_even_transposition_network()

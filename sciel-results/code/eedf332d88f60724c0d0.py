import numpy as np
import random
import time

N = 17

def check_network(comparators):
    """
    Check if the given network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    # Generate all 2^N boolean inputs
    # Using bit manipulation to create the matrix
    # 2^17 = 131072
    # Create a 2D array of shape (2^N, N)
    # inputs[i, j] is the j-th bit of i
    
    # Efficient way to generate inputs
    # We can use np.arange(2**N) and then extract bits
    # But 2**17 is small enough to just loop or use a trick
    
    # Trick: 
    # np.arange(2**N)[:, None] // 2**np.arange(N)[::-1] % 2
    # This creates a (131072, 17) array
    
    inputs = (np.arange(2**N)[:, None] // 2**np.arange(N-1, -1, -1)) % 2
    
    for i, j in comparators:
        # Ensure i < j
        if i > j:
            i, j = j, i
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    # Check if sorted
    # A row is sorted if each element is <= the next
    diffs = np.diff(inputs, axis=1)
    return np.all(diffs >= 0)

def construct():
    # Seed: Odd-even transposition
    seed_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    current_network = seed_network[:]
    best_network = current_network[:]
    
    start_time = time.time()
    timeout = 55  # Leave 5s margin
    
    # Hill Climbing: try to remove comparators
    # Also try to swap or move?
    # For now, just removal and random restarts if stuck
    
    while time.time() - start_time < timeout:
        improved = False
        # Try to remove a random comparator
        if len(current_network) > 0:
            idx = random.randint(0, len(current_network) - 1)
            comp_to_remove = current_network[idx]
            candidate = current_network[:idx] + current_network[idx+1:]
            
            if check_network(candidate):
                current_network = candidate
                best_network = candidate[:]
                improved = True
        
        # If stuck for a while, try a random restart or different strategy?
        # For simplicity, just keep trying removals.
        # To avoid getting stuck in a local minimum where no single removal works,
        # we can occasionally try a random swap.
        
        if not improved and random.random() < 0.1:
            # Try a random swap
            if len(current_network) >= 2:
                idx1 = random.randint(0, len(current_network) - 1)
                idx2 = random.randint(0, len(current_network) - 1)
                if idx1 != idx2:
                    candidate = current_network[:]
                    candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                    if check_network(candidate):
                        current_network = candidate
                        best_network = candidate[:]
                        improved = True

    return best_network

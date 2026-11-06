import numpy as np
import time
import random

N = 18
DEADLINE = 85.0 # 5s margin from 90s limit

def generate_seed_network():
    """Odd-even transposition network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# Pre-compute initial states for verification
# Shape: (2^N, N)
X = np.arange(1 << N, dtype=np.int64)
INITIAL_STATES = np.zeros((1 << N, N), dtype=np.int8)
for i in range(N):
    INITIAL_STATES[:, i] = (X >> i) & 1

def verify_network(net):
    """
    Verify if the network sorts all 2^N boolean inputs.
    Uses pre-allocated initial states and NumPy vectorization.
    """
    # Copy initial states
    states = INITIAL_STATES.copy()
    
    # Apply network
    for i, j in net:
        min_val = np.minimum(states[:, i], states[:, j])
        max_val = np.maximum(states[:, i], states[:, j])
        states[:, i] = min_val
        states[:, j] = max_val
        
    # Check if sorted
    # A sequence is sorted if a[i] <= a[i+1] for all i
    diffs = np.diff(states, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    
    current_network = generate_seed_network()
    best_network = current_network[:]
    best_score = len(current_network)
    
    # Search Strategy:
    # 1. Greedy Removal: Try to remove comparators.
    # 2. Swap Heuristic: If removal fails, try to swap comparators to escape local minima.
    
    # We alternate or prioritize removal.
    # Since removal reduces size, we prioritize it.
    # If no removal is found in a full pass, we allow some swaps.
    
    max_swaps_in_stagnation = 100
    stagnation_swaps = 0
    
    while time.time() < DEADLINE:
        n = len(current_network)
        if n == 0:
            break
            
        # 1. Try to remove a comparator
        # Shuffle indices to avoid bias
        indices = list(range(n))
        random.shuffle(indices)
        
        removed = False
        for idx in indices:
            if time.time() > DEADLINE:
                return best_network
                
            candidate = current_network[:idx] + current_network[idx+1:]
            
            if verify_network(candidate):
                current_network = candidate
                best_network = current_network[:]
                best_score = len(current_network)
                removed = True
                # Restart the loop to re-shuffle and try removing more
                break
                
        if removed:
            stagnation_swaps = 0
            continue
            
        # 2. No removal found. Try to swap to change structure.
        if stagnation_swaps < max_swaps_in_stagnation:
            stagnation_swaps += 1
            if n > 1:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    candidate = current_network[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    if verify_network(candidate):
                        current_network = candidate
                        # Note: best_network is not updated here because size didn't change
                        # But we continue searching from this new state
                        # We do NOT break, we go to the next iteration to try removal again
                        continue
        else:
            # Too many swaps without improvement. 
            # We could reset to best_network and try again, but that might loop.
            # For now, break to save time, or just keep swapping?
            # Let's break to avoid wasting time if stuck deep.
            break
            
    return best_network

import numpy as np
import time
import random

N = 18

def construct():
    # Initial odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    # Shape: (2^18, 18)
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(1 << N) >> i) & 1
        
    def verify(net):
        state = inputs.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        # Check if sorted
        return np.all(state[:, :-1] <= state[:, 1:])
        
    start_time = time.time()
    deadline = 80.0 # Leave 10s margin for 90s limit
    
    # Phase 1: Systematic greedy deletion (backwards)
    # This is more effective than random deletion for finding the "core"
    # We iterate until no more deletions are possible in a full pass
    while time.time() - start_time < deadline:
        improved = False
        # Iterate backwards so that indices remain valid after removal
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time >= deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if verify(candidate):
                network = candidate
                improved = True
                # After a successful deletion, restart the pass
                # to check if further deletions are now possible
                break
        if not improved:
            # No more single deletions possible, move to swaps
            break
            
    # Phase 2: Local search with swaps and random deletions
    # Use random restarts within the local search to escape local optima
    best_network = network[:]
    best_len = len(network)
    
    while time.time() - start_time < deadline:
        # Randomly perturb the current network
        # Try a random swap
        if len(network) > 1:
            idx = random.randint(0, len(network) - 2)
            candidate = network[:idx] + [network[idx+1], network[idx]] + network[idx+2:]
            if verify(candidate):
                network = candidate
                if len(network) < best_len:
                    best_network = network[:]
                    best_len = len(network)
                    
        # Try a random deletion (even if we think it's impossible, 
        # maybe after swaps it becomes possible? Or just retry)
        # Actually, if we did a full pass and found nothing, single deletions 
        # are impossible for the *current* configuration. 
        # But swaps change the configuration.
        
        # Occasionally try a more aggressive move: 
        # Remove a comparator and then try to fix it? No, too complex.
        
        # Try swapping non-adjacent comparators?
        # Sometimes long-range swaps help.
        if len(network) > 2 and random.random() < 0.1:
            i = random.randint(0, len(network) - 1)
            j = random.randint(0, len(network) - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                candidate = network[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if verify(candidate):
                    network = candidate
                    if len(network) < best_len:
                        best_network = network[:]
                        best_len = len(network)

    # Return the best network found
    return best_network

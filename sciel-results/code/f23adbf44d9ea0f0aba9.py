import numpy as np
import time
import random

N = 18

def construct():
    # Initial odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(1 << N) >> i) & 1
        
    def verify(net):
        if not net:
            return False
        state = inputs.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        # Check if sorted
        return np.all(state[:, :-1] <= state[:, 1:])
        
    start_time = time.time()
    deadline = 85.0 # Leave 5s margin for 90s limit
    
    current_network = network
    best_network = network
    best_len = len(best_network)
    
    # Pre-compute some properties?
    # No, just run the search.
    
    def try_remove(net, idx):
        candidate = net[:idx] + net[idx+1:]
        return verify(candidate)
        
    def try_swap(net, idx):
        if idx >= len(net) - 1:
            return False
        candidate = net[:idx] + [net[idx+1], net[idx]] + net[idx+2:]
        return verify(candidate)

    while time.time() - start_time < deadline:
        # Phase 1: Greedy Deletion
        # Randomize the order of checks to avoid bias
        indices = list(range(len(current_network)))
        random.shuffle(indices)
        
        removed_any = False
        for idx in indices:
            if time.time() - start_time > deadline:
                break
            # Map shuffled index to current network index
            # Since we remove elements, indices shift. 
            # It's safer to iterate over a copy of the current list and check validity dynamically.
            # Or, simpler: iterate 0 to len-1, and if we remove one, restart the loop.
            # This is O(N^2) per pass but N is small.
            
            # Let's do a simpler pass:
            # Iterate from 0 to len-1.
            # If remove is valid, remove and break to restart the pass.
            # This ensures we don't skip indices.
            
        # Actually, the "remove and restart" is very effective.
        # Let's implement that.
        
        progress = True
        while progress and time.time() - start_time < deadline:
            progress = False
            for i in range(len(current_network)):
                if time.time() - start_time > deadline:
                    break
                # Try to remove current_network[i]
                candidate = current_network[:i] + current_network[i+1:]
                if verify(candidate):
                    current_network = candidate
                    progress = True
                    break # Restart the for loop
            
        # If no removals, try swapping
        if not progress:
            swapped = False
            # Try random swaps
            for _ in range(100): # Limit swaps per stall to avoid wasting time
                if time.time() - start_time > deadline:
                    break
                if len(current_network) < 2:
                    break
                i = random.randint(0, len(current_network) - 2)
                candidate = current_network[:i] + [current_network[i+1], current_network[i]] + current_network[i+2:]
                if verify(candidate):
                    current_network = candidate
                    swapped = True
                    break # Restart deletion phase
            
            if not swapped:
                # If no swaps work, we might be stuck in a local minimum.
                # We can try a "kick" by inserting a random comparator and then deleting?
                # Or just accept this local minimum.
                # To escape, we can try a different random seed or a more aggressive move.
                # For now, let's just break or try a few more random restarts of the deletion order?
                # The deletion order was deterministic (0 to N).
                # Let's try swapping with a different random seed or just stop.
                # Given time constraints, breaking is safer.
                # But we have time. Let's try a "jitter" move:
                # Remove a random comparator, then try to re-insert it elsewhere?
                # This is complex.
                # Let's just try a few more random swaps with different indices.
                # If still stuck, break.
                pass

        # Update best
        if len(current_network) < best_len:
            best_len = len(current_network)
            best_network = current_network[:]
            
    return best_network

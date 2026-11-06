import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N
_INPUTS = None

def _get_inputs():
    global _INPUTS
    if _INPUTS is None:
        arr = np.arange(NUM_INPUTS, dtype=np.uint32)
        bits = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
        # Extract bits
        # Using shifts
        for i in range(N):
            bits[:, i] = (arr >> (N - 1 - i)) & 1
        _INPUTS = bits
    return _INPUTS

def is_valid(network):
    inputs = _get_inputs()
    state = inputs.copy()
    
    for i, j in network:
        # min to i, max to j
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check sorted
    # Check if state[:, k] <= state[:, k+1] for all k
    # Using diff
    diffs = np.diff(state, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial network: Odd-Even Transposition
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Greedy Deletion
    # We try to remove comparators one by one.
    # We use a list of indices.
    
    # To speed up, we can cache the current valid state?
    # No, state changes with every removal attempt.
    
    # We will iterate through passes.
    # In each pass, we shuffle the indices and try to remove each one.
    # If a removal is successful, we update net and break to restart the pass with new indices.
    # This ensures we don't use stale indices.
    
    max_passes = 20 # Safety limit on passes
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
            
        indices = list(range(len(net)))
        random.shuffle(indices)
        
        removed_any = False
        for idx in indices:
            if time.time() > deadline:
                break
                
            # Try removing net[idx]
            # Create candidate list
            # Optimization: if len(net) is small, this is fast.
            candidate = [net[k] for k in range(len(net)) if k != idx]
            
            if is_valid(candidate):
                net = candidate
                removed_any = True
                break # Restart pass with new indices
        
        if not removed_any:
            # No more single removals possible
            # We can try a few random restarts of the shuffle?
            # Or just stop.
            # Let's do 5 more shuffles to be sure it's not just bad luck.
            for _ in range(5):
                if time.time() > deadline:
                    break
                indices = list(range(len(net)))
                random.shuffle(indices)
                removed_any = False
                for idx in indices:
                    if time.time() > deadline:
                        break
                    candidate = [net[k] for k in range(len(net)) if k != idx]
                    if is_valid(candidate):
                        net = candidate
                        removed_any = True
                        break
                if removed_any:
                    break
            if not removed_any:
                break
                
    return net

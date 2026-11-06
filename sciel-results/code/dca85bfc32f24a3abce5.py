import numpy as np
import time
import random

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # 1. Generate initial network: Batcher's Bitonic Sort
    def batcher_bitonic(n):
        if n == 1:
            return []
        
        def bitonic_sort(start, size, direction):
            if size <= 1:
                return []
            
            half = size // 2
            comps = []
            
            comps += bitonic_sort(start, half, 1)
            comps += bitonic_sort(start + half, half, -1)
            
            comps += bitonic_merge(start, size, direction)
            return comps
        
        def bitonic_merge(start, size, direction):
            if size <= 1:
                return []
            
            half = size // 2
            comps = []
            
            for i in range(half):
                idx1 = start + i
                idx2 = start + i + half
                if direction == 1:
                    comps.append((idx1, idx2))
                else:
                    comps.append((idx2, idx1))
            
            comps += bitonic_merge(start, half, direction)
            comps += bitonic_merge(start + half, half, direction)
            
            return comps
        
        return bitonic_sort(0, n, 1)

    current_net = batcher_bitonic(N)
    
    # Pre-compute inputs
    # Using list comprehension for speed in generation, then np.array
    # 65536 * 16 = 1,048,576 elements.
    inputs_list = [(i >> k) & 1 for i in range(1 << N) for k in range(N)]
    inputs = np.array(inputs_list, dtype=np.int8).reshape(-1, N)
    
    def is_valid_fast(net):
        if not net:
            return False # N=16, empty net is invalid
        state = inputs.copy()
        for i, j in net:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        diffs = state[:, 1:] - state[:, :-1]
        return np.all(diffs >= 0)

    # Verify initial
    if not is_valid_fast(current_net):
        # Fallback
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = current_net
    best_size = len(current_net)
    
    # Search
    # We use a simple hill climbing with swaps to escape local minima
    while time.time() < deadline:
        # Try to improve
        improved = False
        
        # 1. Try Removal
        if len(current_net) > 0:
            # Try a few random removals
            for _ in range(5): # Try 5 random removals per cycle
                idx = random.randint(0, len(current_net) - 1)
                new_net = current_net[:idx] + current_net[idx+1:]
                if is_valid_fast(new_net):
                    current_net = new_net
                    if len(current_net) < best_size:
                        best_size = len(current_net)
                        best_net = current_net
                    improved = True
                    break # Restart loop with new net
        
        # 2. Try Swap (if no improvement or just to explore)
        if not improved and len(current_net) > 1:
            for _ in range(5):
                i = random.randint(0, len(current_net) - 1)
                j = random.randint(0, len(current_net) - 1)
                if i == j:
                    continue
                # Swap
                temp = current_net[i]
                current_net[i] = current_net[j]
                current_net[j] = temp
                
                if is_valid_fast(current_net):
                    # Accept swap, continue searching
                    pass
                else:
                    # Revert
                    current_net[j] = current_net[i]
                    current_net[i] = temp
                    # Break to try removal again? Or just continue?
                    # Let's continue to try other swaps
                    pass
        
    return best_net

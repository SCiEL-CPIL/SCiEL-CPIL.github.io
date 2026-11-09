import time
import numpy as np

def construct():
    N = 17
    start_time = time.time()
    time_limit = 55.0

    def batcher_sort_indices(indices):
        n = len(indices)
        if n == 1:
            return []
        
        comps = []
        even_indices = indices[0::2]
        odd_indices = indices[1::2]
        
        comps.extend(batcher_sort_indices(even_indices))
        comps.extend(batcher_sort_indices(odd_indices))
        
        for i in range(0, n - 1, 2):
            comps.append((indices[i], indices[i+1]))
            
        return comps

    initial_indices = list(range(N))
    network = batcher_sort_indices(initial_indices)
    
    def verify_numpy(net):
        # Generate all 2^N binary inputs
        base = np.arange(2**N, dtype=np.uint32)
        inputs = np.zeros((2**N, N), dtype=np.uint8)
        for i in range(N):
            inputs[:, i] = (base >> i) & 1
            
        # Simulate
        for i, j in net:
            mask = inputs[:, i] > inputs[:, j]
            if np.any(mask):
                inputs[mask, i] = 1 - inputs[mask, i]
                inputs[mask, j] = 1 - inputs[mask, j]
                
        # Check if sorted
        sorted_inputs = np.sort(inputs, axis=1)
        return np.array_equal(inputs, sorted_inputs)

    current_net = list(network)
    
    # Greedy Removal
    i = 0
    while i < len(current_net):
        if time.time() - start_time > time_limit:
            break
        
        temp_net = current_net[:i] + current_net[i+1:]
        
        if verify_numpy(temp_net):
            current_net = temp_net
        else:
            i += 1
            
    return current_net

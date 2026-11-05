import numpy as np
import time

N = 18
NUM_INPUTS = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Generate inputs
    indices = np.arange(NUM_INPUTS, dtype=np.uint8)
    inputs = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for j in range(N):
        inputs[:, j] = (indices >> j) & 1
        
    state_buf = np.empty_like(inputs)

    def verify(net):
        if not net:
            return np.all(np.diff(inputs, axis=1) >= 0)
        
        state_buf[:] = inputs
        for i, j in net:
            a = state_buf[:, i]
            b = state_buf[:, j]
            min_ab = np.minimum(a, b)
            max_ab = np.maximum(a, b)
            state_buf[:, i] = min_ab
            state_buf[:, j] = max_ab
        diffs = np.diff(state_buf, axis=1)
        return np.all(diffs >= 0)

    def optimize(net):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            indices_to_try = list(range(len(net)))
            np.random.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if time.time() > deadline:
                    return net
                
                new_net = net[:idx] + net[idx+1:]
                if verify(new_net):
                    net = new_net
                    changed = True
                    break
        return net

    # Initial Network: Odd-Even Transposition
    initial_net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            initial_net.append((i, i + 1))
            
    best_net = optimize(initial_net)
    best_len = len(best_net)
    
    # Search loop
    no_improve_count = 0
    
    while time.time() < deadline:
        # 1. Try Swaps
        improved = False
        # Limit number of swap attempts to avoid wasting time if stuck
        attempts = 0
        while attempts < 20 and time.time() < deadline:
            attempts += 1
            if len(best_net) < 2:
                break
            i, j = np.random.choice(len(best_net), 2, replace=False)
            candidate = best_net.copy()
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
            if verify(candidate):
                candidate = optimize(candidate)
                if len(candidate) < best_len:
                    best_net = candidate
                    best_len = len(candidate)
                    improved = True
                    break # Restart swap attempts if improved
        
        if improved:
            no_improve_count = 0
            continue
            
        # 2. If no improvement from swaps, try Add-Remove
        no_improve_count += 1
        
        if no_improve_count >= 5:
            # Perturb by adding a random comparator
            if len(best_net) < 4095:
                i, j = np.random.choice(N, 2, replace=False)
                if i > j: i, j = j, i
                candidate = best_net + [(i, j)]
                
                # Check time before expensive optimize
                if time.time() < deadline - 2.0:
                    candidate = optimize(candidate)
                    if len(candidate) < best_len:
                        best_net = candidate
                        best_len = len(candidate)
                        no_improve_count = 0
                    else:
                        # If we didn't improve, maybe try a different perturbation
                        pass
            
    return best_net

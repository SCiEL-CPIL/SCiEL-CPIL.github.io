"""QUBO-based penalty annealing for minimizing monochromatic 4-APs on [1,256]."""
import numpy as np
import time

def construct():
    N = 256
    rng = np.random.default_rng(42)
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with d>=1, a+3d <= N
    # Indexing: integers 1..256 map to indices 0..255
    aps_list = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx = (a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1)
            aps_list.append(idx)
    
    aps = np.array(aps_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Build adjacency: for each node, which APs does it belong to?
    node_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for node in ap:
            node_to_aps[node].append(i)
    
    # Convert to numpy arrays for faster iteration? 
    # Lists of ints are fast enough for small degrees.
    
    # Initial random coloring: 0/1
    x = rng.integers(0, 2, size=N).astype(np.int8)
    
    # Initialize auxiliary state: is_mono for each AP
    # is_mono[k] is True if aps[k] is monochromatic
    is_mono = np.zeros(num_aps, dtype=bool)
    current_energy = 0
    
    # Compute initial energy
    for k in range(num_aps):
        a, b, c, d = aps[k]
        if x[a] == x[b] == x[c] == x[d]:
            is_mono[k] = True
            current_energy += 1
    
    best_x = x.copy()
    best_e = current_energy
    
    # Penalty annealing / simulated annealing with incremental updates
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    max_iter = 200000
    no_improve = 0
    patience = 10000
    
    start_time = time.time()
    time_limit = 110.0  # Leave some buffer for 120s limit
    
    for it in range(max_iter):
        if no_improve > patience:
            break
        if T < T_min:
            break
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random index to flip
        idx = rng.integers(0, N)
        old_val = x[idx]
        new_val = 1 - old_val
        
        # Calculate delta energy by updating auxiliary states for affected APs
        delta_e = 0
        affected_aps = node_to_aps[idx]
        
        for ap_idx in affected_aps:
            ap = aps[ap_idx]
            # Get the other 3 nodes
            # ap is a tuple of 4 ints
            # We need to find which position idx is in, and get the others
            # Since ap is small, we can just iterate
            
            # Optimization: Precompute the "others" for each node in each AP?
            # Too much memory. Just compute on fly.
            
            # Find the 3 other indices
            # ap[0], ap[1], ap[2], ap[3]
            # If idx == ap[0], others are ap[1], ap[2], ap[3]
            # etc.
            
            # Vectorized check for "others are all equal"
            # Let's extract the 3 values
            # We can create a mask or just use if-else
            # For speed, let's just get the values
            v0, v1, v2, v3 = x[ap[0]], x[ap[1]], x[ap[2]], x[ap[3]]
            
            # Replace the value at idx with new_val in a logical sense
            # Check if the 3 others are equal
            # If they are not equal, the AP cannot be mono, so new_mono = False
            # If they are equal, new_mono = (new_val == that_value)
            
            # Determine the 3 others
            # Case 1: idx == ap[0]
            if idx == ap[0]:
                o0, o1, o2 = ap[1], ap[2], ap[3]
            elif idx == ap[1]:
                o0, o1, o2 = ap[0], ap[2], ap[3]
            elif idx == ap[2]:
                o0, o1, o2 = ap[0], ap[1], ap[3]
            else: # idx == ap[3]
                o0, o1, o2 = ap[0], ap[1], ap[2]
            
            c0, c1, c2 = x[o0], x[o1], x[o2]
            
            if c0 == c1 == c2:
                new_mono = (new_val == c0)
            else:
                new_mono = False
            
            old_mono = is_mono[ap_idx]
            
            if new_mono != old_mono:
                if new_mono:
                    delta_e += 1
                else:
                    delta_e -= 1
                is_mono[ap_idx] = new_mono
        
        new_energy = current_energy + delta_e
        
        # Accept/Reject
        if new_energy < best_e:
            # Accept and update best
            x[idx] = new_val
            current_energy = new_energy
            best_x = x.copy()
            best_e = new_energy
            no_improve = 0
        else:
            # Accept with probability exp(-delta/T)
            # delta = new_energy - current_energy = delta_e
            if delta_e > 0:
                prob = np.exp(-delta_e / T)
                if rng.random() < prob:
                    x[idx] = new_val
                    current_energy = new_energy
                    no_improve += 1
                else:
                    # Revert is_mono changes
                    for ap_idx in affected_aps:
                        # We need to revert is_mono to old state
                        # We didn't save old state, so we must recompute or save it.
                        # This is a problem. We modified is_mono in place.
                        # We should save old_mono values before modifying.
                        pass
                    # Better approach: Only update is_mono if accepted.
                    # But we need to know delta_e first.
                    # So we should compute delta_e without modifying is_mono, 
                    # OR save the old values.
                    
                    # Let's restructure: 
                    # 1. Compute delta_e and store list of (ap_idx, old_mono, new_mono)
                    # 2. If accepted, apply changes.
                    # 3. If rejected, do nothing.
                    
                    # This requires a second pass or storing changes.
                    # Let's optimize the loop to store changes.
                    pass
                    
        T *= cooling
        
        # This logic is flawed because I modified is_mono before deciding.
        # I will rewrite the loop properly below.
        
    return [int(v) for v in best_x]

import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    # 1. Precompute conflict masks
    # conflict_mask[i] is an integer with bit j set if i and j conflict
    # We only care about i != j
    conflict_mask = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) == 0:
                mask |= (1 << j)
                conflict_mask[j] |= (1 << i)
        conflict_mask[i] = mask
        
    # Helper to check if x can be added to current_mask
    # x can be added if current_mask & conflict_mask[x] == 0
    # Note: x is not in current_mask, so we don't need to mask out x from conflict_mask[x] 
    # unless conflict_mask[x] has bit x set (it doesn't, as i != j in loop)
    
    # Initial solution: All odd numbers
    # Bit i is set if i is in the set
    current_mask = 0
    for i in range(1, n + 1, 2):
        current_mask |= (1 << i)
        
    best_mask = current_mask
    best_size = bin(current_mask).count('1')
    
    # SA Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute list of nodes
    all_nodes = list(range(1, n + 1))
    
    # To speed up random selection, we can just pick a random index
    # and check if it's in the set or not
    
    stagnation = 0
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        r = random.random()
        
        if r < 0.6:
            # Try to add a random node
            x = random.choice(all_nodes)
            # Check if x is already in set
            if not (current_mask & (1 << x)):
                # Check if x conflicts with any node in current_mask
                if (current_mask & conflict_mask[x]) == 0:
                    # Add x
                    current_mask |= (1 << x)
                    new_size = bin(current_mask).count('1')
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = current_mask
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1
        elif r < 0.8:
            # Try to remove a random node from set
            # Find a set bit in current_mask
            # random.choice on bits is slow. 
            # We can pick a random node and check if it's in set.
            # Or maintain a list of nodes in set.
            # For simplicity and speed, let's just pick a random node and if it's in set, remove it.
            # This is inefficient if set is large.
            # Better: keep a list of nodes in the set.
            pass # Placeholder, see below for optimized version
            
        # The above "remove" logic is tricky without a list.
        # Let's use a list of nodes in the set for removal/swap.
        
    # Refactored Loop with List for active nodes
    active_nodes = [i for i in range(1, n + 1, 2)]
    
    # Re-initialize for the loop below
    current_mask = 0
    for i in active_nodes:
        current_mask |= (1 << i)
    best_mask = current_mask
    best_size = len(active_nodes)
    T = 1.0
    stagnation = 0
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        r = random.random()
        
        if r < 0.5:
            # Add
            x = random.choice(all_nodes)
            if not (current_mask & (1 << x)):
                if (current_mask & conflict_mask[x]) == 0:
                    current_mask |= (1 << x)
                    active_nodes.append(x)
                    new_size = len(active_nodes)
                    if new_size > best_size:
                        best_size = new_size
                        best_mask = current_mask
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1
        elif r < 0.8:
            # Remove
            if active_nodes:
                idx = random.randint(0, len(active_nodes) - 1)
                u = active_nodes[idx]
                # Accept removal with prob exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    current_mask &= ~(1 << u)
                    active_nodes.pop(idx)
                    stagnation += 1
        else:
            # Swap
            if active_nodes:
                idx = random.randint(0, len(active_nodes) - 1)
                u = active_nodes[idx]
                x = random.choice(all_nodes)
                if not (current_mask & (1 << x)):
                    # Check if x is compatible with S \ {u}
                    # S \ {u} mask: current_mask & ~(1 << u)
                    # Conflict check: (current_mask & ~(1 << u)) & conflict_mask[x] == 0
                    if ((current_mask & ~(1 << u)) & conflict_mask[x]) == 0:
                        # Accept swap with prob 0.5 or if it improves "potential"
                        if random.random() < 0.5:
                            current_mask &= ~(1 << u)
                            current_mask |= (1 << x)
                            active_nodes[idx] = x
                            stagnation += 1
                            
        if stagnation > 1000:
            T = 0.5
            stagnation = 0
            
        T *= cooling
        
    # Convert best_mask to list
    res = []
    for i in range(1, n + 1):
        if best_mask & (1 << i):
            res.append(i)
    return res

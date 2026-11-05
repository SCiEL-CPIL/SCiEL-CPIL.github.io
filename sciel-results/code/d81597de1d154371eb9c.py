import random
import time
import numpy as np

def construct():
    best_set = []
    best_size = 0
    start_time = time.time()
    time_limit = 115  # leave some buffer
    
    max_sum = 2000
    # Pre-allocate buffer for sums
    sums_buf = np.empty(1000, dtype=np.int32)
    
    for seed in range(10000):
        if time.time() - start_time > time_limit:
            break
        
        random.seed(seed)
        candidates = list(range(1, 1001))
        
        # Strategy: Mostly random, some sorted
        r = random.random()
        if r < 0.1:
            candidates.sort()
        elif r < 0.2:
            candidates.sort(reverse=True)
        else:
            random.shuffle(candidates)
            
        chosen_list = []
        chosen_np = np.array([], dtype=np.int32)
        counts = np.zeros(max_sum + 1, dtype=np.int8)
        
        for x in candidates:
            # Check 2*x
            if counts[2 * x] >= 2:
                continue
            
            # Check x + y for y in chosen
            n_chosen = len(chosen_list)
            if n_chosen > 0:
                # Fill buffer
                # sums_buf[:n_chosen] = x + chosen_np
                # This creates a temp array x + chosen_np
                # To avoid temp, we can do:
                # chosen_np += x? No, modifies chosen_np.
                # We can use numpy.add(x, chosen_np, out=sums_buf[:n_chosen])
                np.add(x, chosen_np, out=sums_buf[:n_chosen])
                
                # Check max
                if np.max(counts[sums_buf[:n_chosen]]) >= 2:
                    continue
            
            # Accept x
            chosen_list.append(x)
            # Update chosen_np
            # np.append is slow. 
            # We can just keep chosen_list and convert to array only when needed?
            # But we need it for the next iteration's add.
            # Let's try to maintain chosen_np by appending.
            # Or, we can just use the list in the next iteration's add?
            # np.add(x, np.array(chosen_list), ...) -> creates array every time.
            
            # Let's stick to np.append for now, it's C code.
            chosen_np = np.append(chosen_np, x)
            
            counts[2 * x] += 1
            if n_chosen > 0:
                counts[sums_buf[:n_chosen]] += 1
                
        if len(chosen_list) > best_size:
            best_size = len(chosen_list)
            best_set = chosen_list[:]
            
    return best_set

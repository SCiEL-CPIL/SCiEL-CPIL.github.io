import time
import random

def construct():
    # Deadline guard: 50 seconds (safe margin from 60s limit)
    start_time = time.time()
    time_limit = 50.0
    
    # Helper to compute reachable bitset for a set of integers
    # Bit k is 1 if sum k is reachable. We only care about sums <= 200.
    def compute_bits(elements):
        bits = 1  # Sum 0 is reachable
        mask = (1 << 201) - 1  # Bits 0..200
        for x in elements:
            bits |= (bits << x)
            bits &= mask
        return bits

    # 1. Greedy Initialization
    # Try to build a large set. 
    # Elements closer to 100 are "safer" because they contribute less to many small sums? 
    # Actually, large elements are harder to combine to 200.
    # Let's try descending order first.
    
    best_set = set()
    best_bits = 1
    best_size = 0
    
    # Candidate lists
    # We can try multiple greedy orders and pick the best.
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        sorted(range(1, 101), key=lambda x: random.random())
    ]
    
    for order in orders:
        current_set = set()
        current_bits = 1
        mask = (1 << 201) - 1
        for x in order:
            # Check if adding x creates sum 200
            # new_bits = current_bits | (current_bits << x)
            # If bit 200 is set in new_bits, skip.
            new_bits = current_bits | (current_bits << x)
            new_bits &= mask
            if (new_bits >> 200) & 1:
                continue
            current_set.add(x)
            current_bits = new_bits
            
        if len(current_set) > best_size:
            best_set = current_set
            best_bits = current_bits
            best_size = len(current_set)
            
    # 2. Local Search: Additions and Swaps
    # We work with the best set found so far.
    current_set = set(best_set)
    current_bits = best_bits
    current_size = best_size
    
    # Precompute mask
    mask = (1 << 201) - 1
    
    # Loop until no improvement or time runs out
    improved = True
    while improved:
        improved = False
        
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Phase 1: Try to add any element
        # Sort candidates by some heuristic? Random is fine.
        candidates = [x for x in range(1, 101) if x not in current_set]
        random.shuffle(candidates)
        
        added = False
        for x in candidates:
            new_bits = current_bits | (current_bits << x)
            new_bits &= mask
            if (new_bits >> 200) & 1:
                continue
            # Valid addition
            current_set.add(x)
            current_bits = new_bits
            current_size += 1
            improved = True
            added = True
            break # Restart adding phase with new state
        
        if added:
            continue # Go back to start of while loop to try more additions
            
        # Phase 2: Try Swaps if no addition possible
        # If we can't add, we might need to swap to enable an addition later.
        # We accept a swap ONLY if it enables an immediate addition (size +1).
        
        ys = list(current_set)
        xs = [x for x in range(1, 101) if x not in current_set]
        
        random.shuffle(ys)
        random.shuffle(xs)
        
        swap_done = False
        for y in ys:
            if swap_done: break
            for x in xs:
                if time.time() - start_time > time_limit:
                    break
                
                # Perform swap: remove y, add x
                # Compute bits for S' = (S \ {y}) U {x}
                new_set_elements = [e for e in current_set if e != y]
                new_set_elements.append(x)
                
                # Compute bits
                nb = 1
                for e in new_set_elements:
                    nb |= (nb << e)
                    nb &= mask
                    # Early exit if 200 is reached? 
                    if (nb >> 200) & 1:
                        break
                
                if (nb >> 200) & 1:
                    continue # Invalid swap
                
                # Valid swap. Now check if we can ADD to this new state.
                added_z = None
                # Check all z not in new_set_elements
                for z in range(1, 101):
                    if z in new_set_elements:
                        continue
                    nbz = nb | (nb << z)
                    nbz &= mask
                    if (nbz >> 200) & 1 == 0:
                        added_z = z
                        break
                
                if added_z is not None:
                    # Accept Swap + Addition
                    current_set.remove(y)
                    current_set.add(x)
                    current_set.add(added_z)
                    
                    # Update bits
                    current_bits = nbz
                    current_size += 1
                    improved = True
                    swap_done = True
                    break # Break out of x loop, then y loop
                    
            if time.time() - start_time > time_limit:
                break
                
    return list(current_set)

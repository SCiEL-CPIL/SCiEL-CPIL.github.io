import random
import time

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Mask for bits 0 to 200
    mask = (1 << 201) - 1
    
    # Deterministic orders to try first
    orders = [
        list(range(100, 0, -1)), # Largest first
        list(range(1, 101))      # Smallest first
    ]
    
    max_iterations = 5000 # Bitset is very fast
    
    for i in range(max_iterations):
        if i < len(orders):
            order = orders[i]
        else:
            order = list(range(1, 101))
            random.shuffle(order)
        
        selected = []
        reachable_bits = 1 # Bit 0 is set
        
        for x in order:
            # Check if 200-x is reachable
            # If bit (200-x) is set in reachable_bits, then sum 200-x exists.
            # Adding x would create 200.
            if (reachable_bits >> (200 - x)) & 1:
                continue
            
            selected.append(x)
            # Update reachable bits
            # Shift left by x, mask to 200 bits
            new_bits = (reachable_bits << x) & mask
            reachable_bits |= new_bits
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            
        if time.time() - start_time > 85:
            break
            
    return best_set

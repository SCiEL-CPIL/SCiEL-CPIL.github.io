import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    candidates = list(range(2, 41))
    best_size = 0
    best_set = []
    
    # Use a fixed seed for reproducibility? No, random is better for exploration.
    # But if we want to be safe, we can mix strategies.
    
    # Strategy 1: Random restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        current_set = []
        products = {1}
        
        for x in candidates:
            if x in products:
                continue
            
            # Check collisions
            valid = True
            new_prods = []
            for p in products:
                prod = x * p
                if prod in products:
                    valid = False
                    break
                new_prods.append(prod)
            
            if valid:
                current_set.append(x)
                products.update(new_prods)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return best_set

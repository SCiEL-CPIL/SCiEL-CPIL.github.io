import random
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    best_set = []
    best_size = 0
    
    max_n = 1000
    
    def run_greedy(order):
        chosen = []
        counts = {}
        
        for c in order:
            valid = True
            
            # Check 2*c
            if counts.get(2 * c, 0) >= 2:
                valid = False
            
            if valid:
                # Check c + x for x in chosen
                for x in chosen:
                    s = c + x
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
            
            if valid:
                # Update counts
                counts[2 * c] = counts.get(2 * c, 0) + 1
                chosen.append(c)
                for x in chosen[:-1]:
                    s = c + x
                    counts[s] = counts.get(s, 0) + 1
                
        return chosen

    # 1. Deterministic Ascending (Replicates seed behavior, ensures baseline score >= 1.0)
    asc_order = list(range(1, max_n + 1))
    current_set = run_greedy(asc_order)
    best_set = current_set
    best_size = len(current_set)

    # 2. Deterministic Descending
    if time.time() - start_time < deadline:
        desc_order = list(range(max_n, 0, -1))
        current_set = run_greedy(desc_order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # 3. Random Restarts
    # Use a fixed seed for reproducibility in evaluation? No, random is fine.
    # We can also try different random strategies if time allows.
    
    while time.time() - start_time < deadline:
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        current_set = run_greedy(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    return best_set

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer
    
    # 1. Get a strong initial solution using Greedy
    def get_greedy_solution():
        values = []
        products = set()
        for x in range(2, 129):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    best_seq = get_greedy_solution()
    best_len = len(best_seq)
    
    # 2. Branch and Bound Search
    # State: (next_val, current_vals, current_prods)
    # We search for a sequence longer than best_len.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(next_val, current_vals, current_prods):
        nonlocal best_seq, best_len
        
        # Pruning 1: Time limit
        if time.time() - start_time > time_limit:
            return
            
        # Pruning 2: Bound
        # Max possible length = len(current_vals) + (128 - next_val + 1)
        if len(current_vals) + (128 - next_val + 1) <= best_len:
            return
            
        # Pruning 3: Base case
        if next_val > 128:
            if len(current_vals) > best_len:
                best_seq = current_vals
                best_len = len(current_vals)
            return
        
        # Branch 1: Include next_val
        x = next_val
        tails = [x]
        prod = x
        for a in reversed(current_vals):
            prod *= a
            tails.append(prod)
            
        if not current_prods.intersection(tails):
            new_vals = current_vals + [x]
            new_prods = current_prods.union(tails)
            search(next_val + 1, new_vals, new_prods)
            
        # Branch 2: Exclude next_val
        search(next_val + 1, current_vals, current_prods)
        
    # Start search
    search(2, [], set())
    
    return best_seq

# EVOLVE-BLOCK-END

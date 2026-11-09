# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    candidates = list(range(2, 129))
    n = len(candidates)
    
    best_seq = []
    best_len = 0
    
    # 1. Run Greedy to get a good initial lower bound
    def greedy():
        values = []
        products = set()
        for x in candidates:
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    greedy_seq = greedy()
    best_len = len(greedy_seq)
    best_seq = greedy_seq
    
    # 2. Branch and Bound
    # State: idx (current candidate index), current_seq, products_set, suffixes
    # suffixes: list of products of consecutive blocks ending at the last element of current_seq
    
    def search(idx, current_seq, products_set, suffixes):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() - start_time > timeout:
            return
        
        # Update best
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        # Base case
        if idx == n:
            return
            
        # Pruning: Even if we take all remaining, can we beat best?
        if len(current_seq) + (n - idx) <= best_len:
            return
            
        x = candidates[idx]
        
        # Calculate potential new suffixes if we include x
        # Current suffixes: S = [s_1, s_2, ...] where s_1 is last elem, s_2 is last two, etc.
        # New suffixes if add x: T = [x, x*s_1, x*s_2, ...]
        
        t_list = [x]
        prod = x
        for s in suffixes:
            prod *= s
            t_list.append(prod)
            
        # Check if any new suffix is in existing products_set
        if products_set.intersection(t_list):
            # Cannot include x, must exclude
            search(idx + 1, current_seq, products_set, suffixes)
        else:
            # Can include x
            # Branch 1: Include x
            new_products = products_set.union(t_list)
            search(idx + 1, current_seq + [x], new_products, t_list)
            
            # Branch 2: Exclude x
            search(idx + 1, current_seq, products_set, suffixes)

    search(0, [], set(), [])
    
    return best_seq
# EVOLVE-BLOCK-END

import sys
import time

# ... (precomputation same) ...

def construct():
    sys.setrecursionlimit(10000)
    
    best_len = 0
    best_seq = []
    visited_states = set()
    start_time = time.time()
    timeout = 55.0  # Leave 5s for overhead
    
    def dfs(last_val, all_products, suffixes, current_seq, depth):
        nonlocal best_len, best_seq, visited_states
        
        if time.time() - start_time > timeout:
            return
            
        # Pruning
        if len(current_seq) + (128 - last_val) <= best_len:
            return
            
        # Memoization
        if not suffixes:
            key = (last_val, frozenset(all_products), frozenset(suffixes))
        else:
            key = (min(suffixes), frozenset(all_products), frozenset(suffixes))
            
        if key in visited_states:
            return
        visited_states.add(key)
        
        for next_val in range(last_val + 1, 129):
            # Quick check: if time is up, break
            if time.time() - start_time > timeout:
                return
                
            new_suffixes = {next_val * s for s in suffixes}
            new_suffixes.add(next_val)
            
            if all_products.intersection(new_suffixes):
                continue
                
            new_all_products = all_products.union(new_suffixes)
            new_seq = current_seq + [next_val]
            
            if len(new_seq) > best_len:
                best_len = len(new_seq)
                best_seq = new_seq
                
            dfs(next_val, new_all_products, new_suffixes, new_seq, depth + 1)
            
    dfs(1, set(), set(), [], 0)
    
    return best_seq

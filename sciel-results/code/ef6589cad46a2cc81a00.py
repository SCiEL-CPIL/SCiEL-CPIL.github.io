import time

def construct():
    candidates = list(range(2, 41))
    best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    start_time = time.time()
    
    def dfs(idx, current_products, current_selected):
        nonlocal best_set
        if time.time() - start_time > 50:
            return
            
        if idx == len(candidates):
            if len(current_selected) > len(best_set):
                best_set = list(current_selected)
            return
            
        if len(current_selected) + (len(candidates) - idx) <= len(best_set):
            return
            
        x = candidates[idx]
        
        # Try skipping
        dfs(idx + 1, current_products, current_selected)
        
        # Try taking
        conflict = False
        for p in current_products:
            if p * x in current_products:
                conflict = True
                break
        
        if not conflict:
            new_products = current_products | {p * x for p in current_products}
            dfs(idx + 1, new_products, current_selected + [x])

    # Sort candidates to help pruning?
    # Let's try sorting by value.
    candidates.sort()
    
    dfs(0, {1}, [])
    
    return best_set

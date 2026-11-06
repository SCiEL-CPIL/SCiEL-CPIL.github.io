import time

def construct():
    candidates = list(range(2, 129))
    n = len(candidates)
    best_seq = []
    best_len = 0
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s limit
    
    # 1. Greedy Pass to establish a lower bound
    curr_seq = []
    curr_prods = set()
    for x in candidates:
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(curr_seq):
            prod *= a
            tails.append(prod)
        
        if not curr_prods.intersection(tails):
            curr_seq.append(x)
            curr_prods.update(tails)
            
    best_seq = curr_seq.copy()
    best_len = len(best_seq)
    
    # 2. Backtracking Search to maximize length
    def dfs(idx, seq, prods):
        nonlocal best_seq, best_len, start_time, time_limit
        
        # Time check
        if time.time() - start_time > time_limit:
            return
            
        # Update best
        if len(seq) > best_len:
            best_seq = seq.copy()
            best_len = len(seq)
            
        if idx == n:
            return
            
        # Pruning: Max possible length from here is len(seq) + (n - idx)
        if len(seq) + (n - idx) <= best_len:
            return
            
        x = candidates[idx]
        
        # Option 1: Include x
        # Calculate new products ending at x
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(seq):
            prod *= a
            tails.append(prod)
            
        # Check collision with existing products
        if not prods.intersection(tails):
            new_prods = prods.union(tails)
            new_seq = seq + [x]
            dfs(idx + 1, new_seq, new_prods)
            
        # Option 2: Exclude x
        dfs(idx + 1, seq, prods)
        
    dfs(0, [], set())
    
    return best_seq

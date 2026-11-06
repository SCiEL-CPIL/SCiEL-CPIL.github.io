import time

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave some margin
    
    # Greedy initialization
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
            
    best_seq = values
    best_len = len(values)
    
    # DFS to try to extend or improve
    # We will use a recursive function with backtracking
    # To avoid deep recursion issues, we might use a stack or limit depth
    # But depth is likely < 20, so recursion is fine.
    
    def dfs(current_seq, current_sums, depth):
        nonlocal best_seq, best_len, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if len(current_seq) > best_len:
            best_seq = current_seq[:]
            best_len = len(current_seq)
            
        # Try to extend
        if current_seq:
            last_val = current_seq[-1]
        else:
            last_val = 0
            
        # Candidates must be > last_val and <= 256
        # We can try candidates in increasing order, or random order, or specific heuristic
        # Increasing order is what greedy did.
        # Maybe trying larger numbers first helps avoid collisions?
        # Let's try increasing order first, but with a twist:
        # If we fail to extend, we backtrack.
        
        # Optimization: Pre-calculate potential tails?
        # For each candidate x, compute new sums.
        
        for x in range(last_val + 1, 257):
            if time.time() - start_time > time_limit:
                return
                
            tails, total = [x], x
            for a in reversed(current_seq):
                total += a
                tails.append(total)
                
            if not current_sums.intersection(tails):
                new_sums = current_sums.union(tails)
                dfs(current_seq + [x], new_sums, depth + 1)
                
            # If we found a longer sequence, we might stop early?
            # No, we want the longest.
            # But if we just found a valid extension, we should explore it.
            
            # Heuristic: If we've already found a sequence of length L,
            # and we are at depth L, we can stop trying to extend further?
            # No, we want to maximize.
            
            # To speed up, if we have already found a sequence of length >= current_depth + 10,
            # maybe we can prune? No, that's risky.
            
            # A better pruning: If the remaining space (256 - x) is too small to add more elements?
            # Each new element must be at least 1 larger than the previous.
            # So max additional elements is roughly (256 - x) / 1.
            # But we need distinct sums, so they must be larger.
            
            # Let's just let it run until time limit.
            
    dfs(best_seq, sums, 0)
    
    return best_seq

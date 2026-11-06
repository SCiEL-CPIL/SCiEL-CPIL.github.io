import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    best_len = 0
    best_seq = []
    
    def dfs(prefixes, last_s, used_diffs, current_seq):
        nonlocal best_len, best_seq
        
        if time.time() > deadline:
            return
            
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
            
        last_prefix = prefixes[-1]
        
        # We can prune if the theoretical max length from here is <= best_len
        # But estimating that is hard.
        
        for s in range(last_s + 1, 257):
            if time.time() > deadline:
                return
                
            new_prefix = last_prefix + s
            valid = True
            new_diffs = []
            # Check against all existing prefixes
            for p in prefixes:
                diff = new_prefix - p
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
                
            if valid:
                # Optimization: copy set
                # used_diffs is a set
                new_used_diffs = used_diffs.copy()
                new_used_diffs.update(new_diffs)
                
                dfs(prefixes + [new_prefix], s, new_used_diffs, current_seq + [s])

    dfs([0], 0, set(), [])
    
    return best_seq

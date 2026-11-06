import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    time_limit = 50.0
    
    best_seq = []
    
    def dfs(seq, sums, prefix_sums):
        nonlocal best_seq, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            
        if not seq:
            next_min = 1
        else:
            next_min = seq[-1] + 1
            
        if next_min > 256:
            return
            
        # Generate candidates
        # Bias towards smaller numbers? Or random?
        # Random is safer for exploration.
        candidates = list(range(next_min, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                return
                
            # Check conflict
            total = prefix_sums[-1] + x
            new_sums = []
            conflict = False
            for p in prefix_sums:
                s = total - p
                if s in sums:
                    conflict = True
                    break
                new_sums.append(s)
                
            if conflict:
                continue
                
            # Update state
            new_sums_set = set(new_sums)
            sums.update(new_sums_set)
            new_prefix = prefix_sums + [total]
            seq.append(x)
            
            dfs(seq, sums, new_prefix)
            
            # Backtrack
            seq.pop()
            for s in new_sums_set:
                sums.remove(s)

    # Run multiple restarts
    # First run with a fixed seed for reproducibility? No, random is fine.
    while time.time() - start_time < time_limit - 1.0:
        dfs([], set(), [0])
        
    return best_seq

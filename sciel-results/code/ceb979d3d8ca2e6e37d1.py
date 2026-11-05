import sys
import time

sys.setrecursionlimit(10000)

def construct():
    target_diffs = set(range(1, 31))
    
    # 1. Greedy to get initial bound
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        # Try adding x such that x - s = d or s - x = d (but x > s usually in sorted add)
        # Actually, just try small candidates
        candidates = set()
        for v in a:
            candidates.add(v + d)
            candidates.add(v - d)
        for x in sorted(candidates):
            if x < 0 or x in a:
                continue
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            # Fallback for greedy: add large number
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    
    best_max = max(a)
    best_set = a[:]
    
    # 2. DFS to improve
    # Time limit guard
    start_time = time.time()
    time_limit = 140  # 120s limit, keep 20s margin
    
    current_set = [0]
    used_diffs = set()
    
    def dfs(s, used_diffs):
        nonlocal best_max, best_set, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        missing = target_diffs - used_diffs
        if not missing:
            max_val = s[-1]
            if max_val < best_max:
                best_max = max_val
                best_set = s[:]
            return
            
        if s[-1] >= best_max:
            return
            
        k = len(s)
        rem_slots = 32 - k
        if rem_slots <= 0:
            return
            
        capacity = rem_slots * k + rem_slots * (rem_slots - 1) // 2
        if len(missing) > capacity:
            return
            
        start_x = s[-1] + 1
        end_x = best_max
        
        # Iterate x
        for x in range(start_x, end_x):
            valid = True
            new_diffs = []
            for s_i in s:
                d = x - s_i
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            new_s = s + [x]
            new_used_diffs = used_diffs.union(new_diffs)
            
            dfs(new_s, new_used_diffs)
            
            # If we found a solution with max < current best, end_x might shrink in next loop iterations?
            # No, end_x is fixed for this call. But best_max updates.
            # We can break if best_max == start_x? No.
            
    dfs(current_set, used_diffs)
    
    return best_set

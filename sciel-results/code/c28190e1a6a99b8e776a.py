import time
import sys

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    best_max = float('inf')
    best_set = None
    
    # used_small: bitmask for diffs 1..30
    # used_large: set for diffs > 30
    # current: list of sorted elements
    
    def dfs(current, used_small, used_large):
        nonlocal best_max, best_set
        nonlocal deadline, start_time
        
        if time.time() - start_time > deadline:
            return
            
        max_val = current[-1]
        
        # Check if all target diffs are covered
        if used_small == full_mask:
            if max_val < best_max:
                best_max = max_val
                best_set = list(current)
            return
            
        # Prune if current max is already not better than best found
        if max_val >= best_max:
            return
            
        # If we have too many elements, prune
        if len(current) >= 32:
            return
            
        last = current[-1]
        
        # Upper bound for x
        upper_bound = best_max - 1 if best_max < float('inf') else 500
        
        for x in range(last + 1, upper_bound + 1):
            if time.time() - start_time > deadline:
                return

            valid = True
            new_diffs_small_mask = 0
            new_diffs_large = []
            
            for s in current:
                d = x - s
                if d <= 30:
                    bit = 1 << (d - 1)
                    if used_small & bit:
                        valid = False
                        break
                    new_diffs_small_mask |= bit
                else:
                    if d in used_large:
                        valid = False
                        break
                    new_diffs_large.append(d)
                    
            if not valid:
                continue
                
            # If this x completes the set, update best and break
            if (used_small | new_diffs_small_mask) == full_mask:
                if x < best_max:
                    best_max = x
                    best_set = current + [x]
                break # No larger x can be better in this branch

            # Recurse
            current.append(x)
            old_used_small = used_small
            used_small = used_small | new_diffs_small_mask
            for d in new_diffs_large:
                used_large.add(d)
            
            dfs(current, used_small, used_large)
            
            # Backtrack
            current.pop()
            used_small = old_used_small
            for d in new_diffs_large:
                used_large.remove(d)
            
            # Prune subsequent x if x >= best_max
            if x >= best_max:
                break

    # Initial state
    initial_set = [0]
    initial_used_small = 0
    initial_used_large = set()
    
    dfs(initial_set, initial_used_small, initial_used_large)
    
    if best_set:
        return best_set
    else:
        # Fallback
        S = [0]
        covered = set()
        while not covered.issuperset(target_diffs) and len(S) <= 32:
            found = False
            for d in target_diffs - covered:
                for s in S:
                    c = s + d
                    if c in S:
                        continue
                    new_S = S + [c]
                    new_S.sort()
                    diffs = set(y - x for i, x in enumerate(new_S) for y in new_S[i+1:])
                    if len(diffs) == len(new_S) * (len(new_S) - 1) // 2:
                        S = new_S
                        covered = diffs
                        found = True
                        break
                if found:
                    break
            if not found:
                c = 2 * max(S) + 3
                while True:
                    new_S = S + [c]
                    new_S.sort()
                    diffs = set(y - x for i, x in enumerate(new_S) for y in new_S[i+1:])
                    if len(diffs) == len(new_S) * (len(new_S) - 1) // 2:
                        S = new_S
                        covered = diffs
                        break
                    c += 1
        return S

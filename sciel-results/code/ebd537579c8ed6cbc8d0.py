import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    target_diffs = set(range(1, 31))
    target_mask = 0
    for d in target_diffs:
        target_mask |= (1 << d)
        
    # We need at least 9 elements
    # Let's try k=9 first
    # If k=9 fails within time, try k=10?
    # But k=9 should be sufficient and likely optimal for M.
    
    def solve(M, k=9):
        # Returns a valid set or None
        # Backtracking
        # State: current_set (list), used_mask (int), covered_mask (int)
        
        # Optimization: 
        # We must include 0.
        # We need to choose k-1 more elements from 1..M.
        
        # Pruning:
        # If number of remaining elements * (k - current_size) < remaining target diffs?
        # Not simple because diffs overlap.
        
        def backtrack(idx, current_set, used_mask, covered_mask):
            if time.time() > deadline:
                return None
            
            if len(current_set) == k:
                if (covered_mask & target_mask) == target_mask:
                    return current_set[:]
                return None
            
            # Heuristic: 
            # We can try to fill gaps.
            # But simple iteration might be enough with pruning.
            
            # Lower bound for next element:
            last = current_set[-1]
            # We can start from last + 1
            
            # Upper bound: M
            
            # Additional pruning:
            # If we have r elements left to place, max new diffs is r * len(current_set) + r*(r-1)/2
            # Total diffs so far + max new diffs >= 30?
            # Actually, we just need to cover 1..30.
            
            # Let's just iterate
            for x in range(last + 1, M + 1):
                # Check Sidon property
                valid = True
                new_mask = 0
                for y in current_set:
                    d = x - y
                    if d > M: # Not possible since x<=M, y>=0
                        continue
                    if (used_mask >> d) & 1:
                        valid = False
                        break
                    new_mask |= (1 << d)
                
                if not valid:
                    continue
                
                new_used_mask = used_mask | new_mask
                new_covered_mask = covered_mask | (new_mask & target_mask)
                
                # Pruning: If we can't cover all targets with remaining elements
                # Max additional coverage:
                # Remaining slots: k - len(current_set) - 1
                # This is complex to bound tightly.
                
                res = backtrack(idx + 1, current_set + [x], new_used_mask, new_covered_mask)
                if res is not None:
                    return res
            
            return None
        
        return backtrack(0, [0], 0, 0)

    # Try increasing M
    # Lower bound for M:
    # We need 30 diffs.
    # Max diff is M.
    # So M >= 30.
    
    for M in range(30, 100):
        if time.time() > deadline:
            break
        res = solve(M)
        if res is not None:
            return res
    
    # Fallback: Greedy from original code if search fails or times out
    # But search should find it.
    # If M goes up to 100 and no solution, something is wrong.
    # A perfect difference set of size 8 has M=55.
    # But size 8 is not enough diffs.
    # Size 9 should have M around 50-60.
    
    # If we reach here, return a valid (though maybe suboptimal) set
    # Re-run greedy as fallback
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

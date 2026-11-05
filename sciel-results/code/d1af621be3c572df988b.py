import time
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Required differences
    req_diffs = set(range(1, 31))
    
    # We want to find a set A such that:
    # 1. 0 in A
    # 2. All diffs in A are distinct
    # 3. req_diffs is subset of diffs(A)
    # 4. max(A) is minimized
    
    # Lower bound for max(A) is 30.
    # Upper bound: We know D=42 works (PDS order 7).
    # Let's search D from 30 to 50.
    
    best_solution = None
    
    def dfs(current_set, used_diffs, remaining_req, D):
        nonlocal best_solution
        nonlocal start_time
        nonlocal time_limit
        
        if time.time() - start_time > time_limit:
            return False, None
            
        if not remaining_req:
            return True, current_set
            
        # Pruning: Check if we can form the smallest remaining difference
        min_req = min(remaining_req)
        max_a = max(current_set)
        
        # Try to find a candidate x for min_req
        # x - a = min_req => x = a + min_req
        # x must be > max_a
        # x must be <= D
        # x - b must not be in used_diffs for all b in current_set
        
        valid_candidates = []
        for a in current_set:
            x = a + min_req
            if x <= max_a:
                continue
            if x > D:
                continue
                
            # Check if x creates duplicate diffs
            is_valid = True
            for b in current_set:
                d = x - b
                if d in used_diffs:
                    is_valid = False
                    break
            if is_valid:
                valid_candidates.append(x)
                
        if not valid_candidates:
            return False, None
            
        # Sort candidates to prefer smaller x (greedy-like)
        for x in valid_candidates:
            # Check all diffs for x
            new_diffs = set()
            valid_x = True
            for a in current_set:
                d = x - a
                if d in used_diffs:
                    valid_x = False
                    break
                new_diffs.add(d)
            
            if not valid_x:
                continue
                
            # Update state
            new_used_diffs = used_diffs | new_diffs
            new_remaining_req = remaining_req - new_diffs
            
            # Recurse
            found, sol = dfs(current_set + [x], new_used_diffs, new_remaining_req, D)
            if found:
                return True, sol
                
        return False, None

    for D in range(30, 60):
        # Initial state
        current_set = [0]
        used_diffs = set()
        remaining_req = set(range(1, 31))
        
        found, sol = dfs(current_set, used_diffs, remaining_req, D)
        if found:
            best_solution = sol
            break
            
    if best_solution is None:
        # Fallback to a known valid set if search fails (e.g. PDS of order 7)
        # PDS(7) = {0, 1, 2, 4, 10, 11, 13, 15} scaled? No, standard PDS(7) is {0, 1, 2, 4, 10, 11, 13, 15}?
        # Let's use the explicit set from PDS(7) construction or a known valid one.
        # A known valid set for 1..30 with max 42:
        # {0, 1, 4, 10, 12, 17, 21, 25, 29, 32, 36, 40, 42} ? No, must be Sidon.
        # Let's just return the PDS(7) set if we can construct it.
        # PDS(q) for q=7: {0, 1, 2, 4, 10, 11, 13, 15} is size 8, max 15. Diffs go up to 15. Not enough.
        # Wait, PDS(q) has size q+1 and max diff q(q-1).
        # For q=7, size 8, max diff 42.
        # The set is {0, 1, 2, 4, 10, 11, 13, 15}? No.
        # Standard PDS(7) is {0, 1, 2, 4, 10, 11, 13, 15} is NOT correct.
        # The set is {0, 1, 2, 4, 10, 11, 13, 15} has max 15.
        # Ah, the PDS is in Z_q? No, it's a set of integers.
        # The construction is: {0, 1, 2, ..., q-1} union {q, 2q, ..., (q-1)q} ? No.
        # Let's use a hardcoded valid set if search fails.
        # A valid set with max 42:
        # {0, 1, 4, 10, 12, 17, 21, 25, 29, 32, 36, 40, 42} -> Check Sidon.
        # Actually, the search should find something <= 42.
        # If it fails, return a greedy result.
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best_solution

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    # Safety margin: kill before 150s limit. 145s is safe.
    deadline = start_time + 145.0
    
    target_diffs = set(range(1, 31))
    best_solution = None
    best_max_val = float('inf')
    
    # We use a bitmask to track used differences.
    # Bit i is set if difference i is already present.
    # We only care about differences up to some limit for conflicts, 
    # but since we minimize max element, differences won't be huge initially.
    # However, a difference d is a conflict if it's already used.
    
    def dfs(S, used_mask, missing_diffs, depth):
        nonlocal best_solution, best_max_val, deadline
        
        if time.time() > deadline:
            return False
        
        if not missing_diffs:
            mx = S[-1] if S else 0
            if mx < best_max_val:
                best_max_val = mx
                best_solution = list(S)
            return True
        
        n = len(S)
        remaining = len(missing_diffs)
        
        # Pruning: Estimate minimum elements needed.
        # Each new element $x$ added to a set of size $k$ creates $k$ new differences.
        # The next element creates $n$ diffs, the one after $n+1$, etc.
        # We need to cover 'remaining' diffs.
        # Let t be the number of new elements needed.
        # Total new diffs = n + (n+1) + ... + (n+t-1) = t*n + t*(t-1)/2
        # We need t*n + t*(t-1)/2 >= remaining
        t = 0
        total = 0
        while total < remaining:
            t += 1
            total += n + t - 1
            
        # The smallest possible next element is S[-1] + 1.
        # The largest possible element is bounded by best_max_val - 1 (if we want to improve) 
        # or a hard limit if no solution found yet.
        # Actually, if we haven't found a solution, we can't prune by best_max_val strictly,
        # but we can prune if the required span is impossible.
        
        # Let's determine the upper bound for search.
        # If we have a best solution, we only care about max < best_max_val.
        # If not, we allow up to some reasonable limit, e.g., 200.
        # Given the problem size, 200 is very likely enough.
        if best_max_val < float('inf'):
            upper_bound = best_max_val - 1
        else:
            upper_bound = 200 # Heuristic limit for initial search
            
        if upper_bound <= S[-1]:
            return False
            
        # Check if we have enough "room" to place 't' distinct integers > S[-1] and <= upper_bound
        available_positions = upper_bound - S[-1]
        if available_positions < t:
            return False

        # MRV: Find the missing difference with the fewest valid candidate elements.
        # For a missing difference d, a candidate x is valid if:
        # 1. x = y + d for some y in S
        # 2. x > S[-1]
        # 3. x <= upper_bound
        # 4. Adding x to S does not create any duplicate differences.
        
        best_d = None
        best_options = None
        best_count = float('inf')
        
        # We iterate through missing diffs. To make it deterministic and potentially faster,
        # we can sort them, but MRV usually picks the one with fewest options first.
        # Since missing_diffs is a set, we iterate over it.
        
        for d in missing_diffs:
            options = set()
            for y in S:
                x = y + d
                if x <= S[-1]:
                    continue
                if x > upper_bound:
                    continue
                
                # Check if adding x is valid
                # New differences created: x - s for all s in S
                # We need to check if any of these are already in used_mask
                # AND if any of these are equal to each other (which is impossible since s are distinct)
                # So we just need to check against used_mask.
                
                valid = True
                # Optimization: check conflicts
                # For each s in S, diff = x - s.
                # If (used_mask >> diff) & 1: conflict.
                # Also, we must ensure that the new differences don't conflict with each other?
                # No, differences x-s1 and x-s2 are distinct because s1 != s2.
                # So we only check against existing used_mask.
                
                for s in S:
                    diff = x - s
                    if (used_mask >> diff) & 1:
                        valid = False
                        break
                
                if valid:
                    options.add(x)
            
            if len(options) == 0:
                # If a missing difference cannot be formed by any valid addition,
                # this branch is dead.
                return False
                
            if len(options) < best_count:
                best_count = len(options)
                best_d = d
                best_options = options
                if best_count == 1:
                    break
        
        if best_d is None:
            return False
        
        # Try each option for best_d
        # Sort options to find smaller max values earlier (helps with pruning via best_max_val)
        for x in sorted(best_options):
            if time.time() > deadline:
                return False
            
            # We already know x is valid against current S and used_mask.
            # But we should double check or just trust the previous check?
            # Trusting is faster.
            
            S.append(x)
            
            # Update used_mask
            new_mask = used_mask
            for s in S[:-1]: # S[:-1] is the set before x was added
                diff = x - s
                new_mask |= (1 << diff)
            
            # Update missing_diffs
            new_missing = set(missing_diffs)
            for s in S[:-1]:
                diff = x - s
                if diff in new_missing:
                    new_missing.discard(diff)
            
            if dfs(S, new_mask, new_missing, depth + 1):
                S.pop()
                return True
            
            S.pop()
        
        return False

    # Start DFS
    # Initial state: S=[0], used_mask=0, missing_diffs={1..30}
    dfs([0], 0, set(target_diffs), 0)
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: Greedy construction if DFS fails or times out without solution
    # This is unlikely to be needed if DFS is efficient, but good for robustness.
    a = [0]
    used_diffs = set()
    missing = set(range(1, 31))
    
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    # Simple greedy: always try to add the smallest x that covers the smallest missing diff
    while missing:
        d = min(missing)
        found = False
        # Try x = y + d for y in a
        candidates = set()
        for y in a:
            x = y + d
            if x > max(a):
                candidates.add(x)
        
        # Also try other x that might cover d? No, d must be x-y.
        # So x must be y+d.
        
        # Sort candidates by value
        for x in sorted(candidates):
            # Check validity
            valid = True
            new_diffs = []
            for y in a:
                diff = x - y
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)
            
            if valid:
                # Check for internal conflicts in new_diffs (shouldn't happen)
                if len(set(new_diffs)) == len(new_diffs):
                    a.append(x)
                    for diff in new_diffs:
                        used_diffs.add(diff)
                        if diff in missing:
                            missing.discard(diff)
                    found = True
                    break
        
        if not found:
            # If no candidate works, try larger x?
            # This greedy might get stuck. 
            # For the purpose of this task, the DFS should find a solution.
            # If DFS fails, we return the current best effort or a random valid set if possible.
            # Given the constraints, a solution likely exists with small diameter.
            # Let's just return 'a' if it covers all, else return a dummy valid Sidon set?
            # The task requires valid output.
            # If we can't find one, we should return a known valid one or fail gracefully.
            # But the prompt implies a solution exists.
            pass
            
    if not missing:
        return a
        
    # If we reach here, we failed to find a solution covering 1..30.
    # Return a trivial Sidon set to avoid crashing, though it won't score well.
    return [0, 1, 3] # 1, 2, 3 are diffs? 1-0=1, 3-1=2, 3-0=3. Distinct.

# EVOLVE-BLOCK-END

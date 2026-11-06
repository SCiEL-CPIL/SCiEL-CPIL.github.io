import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave 20s margin for 120s limit
    
    best_solution = None
    best_max_val = float('inf')
    
    # Fallback: Seed program logic
    def seed_construct():
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

    # Initial state
    initial_S = (0,)
    initial_D_used = set()
    initial_D_missing = set(range(1, 31))
    
    def dfs(S, D_used, D_missing):
        nonlocal best_solution, best_max_val
        
        # Check time
        if time.time() > deadline:
            return False
            
        # Check if solved
        if not D_missing:
            curr_max = max(S)
            if curr_max < best_max_val:
                best_max_val = curr_max
                best_solution = list(S)
            return True # Found a solution, but keep searching for better? 
                        # Actually, we want to explore other branches too.
                        # But if we found a solution, we can prune branches 
                        # with max(S) >= best_max_val.
            
        # Prune if current max is already worse than best
        if max(S) >= best_max_val:
            return False
            
        # Prune if size too large
        if len(S) >= 32:
            return False
            
        # Find valid extensions for each missing difference
        # We want to pick the most constrained d
        candidates_for_d = {}
        
        for d in D_missing:
            valid_xs = []
            for y in S:
                x = y + d
                # x must be > max(S) ? Not necessarily, but if x < max(S), 
                # it would have been in S or created a diff earlier.
                # Since S is sorted and we only add new elements, 
                # and we check D_used, x could be smaller than max(S)?
                # If x < max(S), then x is not in S (else diff existed).
                # But if x < max(S), then max(S) - x is a diff.
                # If max(S) - x is in D_used, collision.
                # If not, it's a new diff.
                # However, usually we add elements to the right to keep max small?
                # No, we want to minimize max.
                # Let's allow x anywhere >= 0.
                # But S is a set of distinct integers.
                # If x is not in S, it's a new element.
                
                # Check Sidon condition:
                # All differences x - s for s in S must not be in D_used.
                # And x - s must be positive? 
                # The task says "positive differences".
                # So we consider |x - s|.
                # If x > s, diff = x - s.
                # If x < s, diff = s - x.
                
                # Efficient check:
                # We need to ensure that for all s1, s2 in S U {x}, |s1 - s2| are distinct.
                # Existing diffs in S are distinct (invariant).
                # New diffs involve x.
                # They are { |x - s| for s in S }.
                # These must be distinct from each other (they are, if S has distinct elements).
                # They must be distinct from D_used.
                
                is_valid = True
                for s in S:
                    diff = abs(x - s)
                    if diff in D_used:
                        is_valid = False
                        break
                if is_valid:
                    valid_xs.append(x)
            
            if valid_xs:
                # Remove duplicates? x is determined by y+d. 
                # Different y can give same x? 
                # y1 + d = y2 + d => y1 = y2. So unique.
                candidates_for_d[d] = valid_xs
        
        if not candidates_for_d:
            # No valid single element extension for any missing d
            # We might need to add 2 elements, but let's prune this branch
            # for now to keep it fast. 
            # If this happens often, we might miss solutions.
            # However, with 30 diffs and max size 32, we have room.
            return False
        
        # Select d with fewest candidates
        # Tie-breaker: smallest d?
        target_d = min(candidates_for_d.keys(), key=lambda d: (len(candidates_for_d[d]), d))
        candidates = candidates_for_d[target_d]
        
        # Sort candidates by resulting max(S)
        # To minimize max, we prefer smaller max(S U {x})
        # max(S U {x}) = max(max(S), x)
        # So sort by x.
        candidates.sort()
        
        for x in candidates:
            new_S = tuple(sorted(S + (x,)))
            new_D_used = set(D_used)
            new_D_missing = set(D_missing)
            
            # Update D_used and D_missing
            for s in S:
                diff = abs(x - s)
                new_D_used.add(diff)
                if diff in new_D_missing:
                    new_D_missing.remove(diff)
            
            # Recurse
            dfs(new_S, new_D_used, new_D_missing)
            
            # Check time again
            if time.time() > deadline:
                return False
        
        return False

    # Start DFS
    dfs(initial_S, initial_D_used, initial_D_missing)
    
    if best_solution is None:
        # Fallback to seed
        best_solution = seed_construct()
        
    return best_solution

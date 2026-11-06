import time
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    target_diffs = set(range(1, 31))
    # We know we need at least 9 elements (36 diffs) to cover 30 diffs without overlap.
    # Actually, min size k for Sidon set covering 1..N is roughly sqrt(2N).
    # For N=30, k ~ 8. 8 elements -> 28 diffs. Not enough.
    # 9 elements -> 36 diffs. Enough.
    # So we expect a solution of size 9 or 10.
    
    best_solution = None
    best_max_val = float('inf')
    
    # DFS state
    # S: sorted list of elements
    # used_diffs: set of differences already present
    # missing_diffs: set of required differences not yet covered
    
    def dfs(S, used_diffs, missing_diffs):
        nonlocal best_solution, best_max_val, deadline
        
        if time.time() > deadline:
            return None
            
        if not missing_diffs:
            # Found a solution
            if S and max(S) < best_max_val:
                best_max_val = max(S)
                best_solution = list(S)
            return True
            
        # Pruning: Check if remaining capacity is sufficient
        # Current size k = len(S)
        # If we stop here, we have 0 more elements.
        # We need to cover len(missing_diffs) diffs.
        # Each new element x adds len(S) new differences.
        # Subsequent elements add more.
        # Simple bound: If we add m elements, max new diffs = m*len(S) + m*(m-1)/2.
        # We need this >= len(missing_diffs).
        # Also, the new diffs must be distinct from used_diffs and each other.
        
        # Generate candidates
        # x must be such that x - y is in missing_diffs for some y in S
        # So x = y + d for y in S, d in missing_diffs
        candidates = set()
        max_S = S[-1] if S else -1
        
        for y in S:
            for d in missing_diffs:
                x = y + d
                if x > max_S:
                    candidates.add(x)
        
        # Sort candidates to try smaller ones first (helps minimize diameter)
        for x in sorted(candidates):
            # Check if x creates duplicate differences
            valid = True
            new_diffs = []
            for y in S:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            # Check if new_diffs have duplicates among themselves
            if len(set(new_diffs)) != len(new_diffs):
                continue
            
            # Prepare new state
            new_S = S + [x]
            new_used_diffs = used_diffs.union(new_diffs)
            new_missing_diffs = missing_diffs - set(new_diffs)
            
            # Recurse
            if dfs(new_S, new_used_diffs, new_missing_diffs):
                return True
                
    # Initial call
    # S = [0]
    # used_diffs = set()
    # missing_diffs = target_diffs
    dfs([0], set(), target_diffs)
    
    if best_solution is None:
        # Fallback: Return a valid but potentially large set if DFS failed or timed out
        # This shouldn't happen for N=30, but just in case.
        # We can use the greedy from the seed program as a fallback
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
        
    return best_solution

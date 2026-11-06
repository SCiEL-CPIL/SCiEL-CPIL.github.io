"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    target_diffs = set(range(1, 31))
    best_solution = None
    best_max_val = float('inf')
    
    # Precompute: for each candidate x (1..80) and each existing element y,
    # the difference x - y. We'll use bitmasks for speed.
    # Actually, let's use a different approach:
    # We build the set element by element, but use MRV on the differences.
    
    def get_candidates(S, missing_diffs, used_diffs, upper_bound):
        """Generate candidate elements that could help cover missing diffs."""
        candidates = set()
        max_S = S[-1] if S else 0
        for y in S:
            for d in missing_diffs:
                x = y + d
                if x > max_S and x <= upper_bound:
                    candidates.add(x)
        return candidates
    
    def is_valid_addition(x, S, used_diffs):
        """Check if x can be added to S without creating duplicate differences."""
        new_diffs = []
        for y in S:
            d = x - y
            if d in used_diffs:
                return False, None
            new_diffs.append(d)
        if len(set(new_diffs)) != len(new_diffs):
            return False, None
        return True, new_diffs
    
    def dfs(S, used_diffs, missing_diffs, depth):
        nonlocal best_solution, best_max_val, deadline
        
        if time.time() > deadline:
            return False
        
        if not missing_diffs:
            mx = S[-1] if S else 0
            if mx < best_max_val:
                best_max_val = mx
                best_solution = list(S)
            return True
        
        # Pruning: check if we can cover remaining diffs
        n = len(S)
        remaining = len(missing_diffs)
        # Each new element adds at most n new useful diffs (to existing elements)
        # We need 'remaining' more diffs
        # Minimum elements needed: solve for t where n + (n+1) + ... + (n+t-1) >= remaining
        # That's t*n + t*(t-1)/2 >= remaining
        t = 0
        total = 0
        while total < remaining:
            t += 1
            total += n + t - 1
        # We need at least t more elements, each must be > S[-1]
        # And at most upper_bound - S[-1] positions available
        upper_bound = best_max_val - 1
        if upper_bound <= S[-1]:
            return False
        available_positions = upper_bound - S[-1]
        if available_positions < t:
            return False
        
        # MRV: find the missing difference with fewest valid placements
        best_d = None
        best_options = None
        best_count = float('inf')
        
        for d in sorted(missing_diffs):
            options = set()
            for y in S:
                x = y + d
                if x > S[-1] and x <= upper_bound:
                    valid, _ = is_valid_addition(x, S, used_diffs)
                    if valid:
                        options.add(x)
            if len(options) == 0:
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
        for x in sorted(best_options):
            if time.time() > deadline:
                return False
            
            valid, new_diffs = is_valid_addition(x, S, used_diffs)
            if not valid:
                continue
            
            S.append(x)
            added = set()
            for d in new_diffs:
                used_diffs.add(d)
                added.add(d)
                if d in missing_diffs:
                    missing_diffs.discard(d)
            
            if dfs(S, used_diffs, missing_diffs, depth + 1):
                # Undo
                S.pop()
                for d in added:
                    used_diffs.discard(d)
                return True
            
            # Undo
            S.pop()
            for d in added:
                used_diffs.discard(d)
        
        return False
    
    # Initial DFS
    dfs([0], set(), set(target_diffs), 0)
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy construction
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for _ in range(100):
        covered = set(get_diffs(sorted(a)))
        missing = [d for d in range(1, 31) if d not in covered]
        if not missing:
            break
        d = missing[0]
        found = False
        for x in range(1, 10000):
            if x in a:
                continue
            trial = sorted(a + [x])
            diffs = get_diffs(trial)
            if len(set(diffs)) == len(diffs) and d in diffs:
                a = trial
                found = True
                break
        if not found:
            x = max(a) + d + 1
            while x in a or any((x - ai) in get_diffs(a) for ai in a):
                x += 1
            a = sorted(a + [x])
    
    return a
# EVOLVE-BLOCK-END

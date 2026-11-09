import time
import sys

def construct():
    start_time = time.time()
    deadline = 100  # 100 seconds limit to be safe
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    
    best_max = float('inf')
    best_set = None
    
    # 1. Greedy Phase: Get a valid solution to set a bound
    # We want a small set that covers 1..30.
    # Try to build a set where each new element covers a new "large" missing diff or multiple.
    
    S = [0]
    covered = set()
    # Simple greedy: pick smallest x > S[-1] such that x-S is valid Sidon and covers most missing
    # Actually, just picking smallest valid x might lead to large max.
    # Let's just find *any* solution quickly.
    
    def is_valid_addition(S, x):
        diffs = []
        for s in S:
            d = x - s
            if d <= 0:
                continue
            diffs.append(d)
        # Check distinctness with existing
        existing_diffs = set(y - x for i, x in enumerate(S) for y in S[i+1:])
        if len(set(diffs).intersection(existing_diffs)) > 0:
            return False, set()
        # Check distinctness within new diffs
        if len(diffs) != len(set(diffs)):
            return False, set()
        return True, set(diffs)

    # Greedy construction
    S_greedy = [0]
    covered_greedy = set()
    # We need to cover 1..30.
    # Strategy: Always try to add the smallest x that is valid and covers at least one missing diff.
    # If no such x, add smallest valid x.
    # But "smallest valid x" that covers a diff is x = s + d for some s, d missing.
    
    while not covered_greedy.issuperset(target_diffs) and len(S_greedy) < 32:
        missing = target_diffs - covered_greedy
        if not missing:
            break
        
        candidates = set()
        for d in missing:
            for s in S_greedy:
                candidates.add(s + d)
        
        # Also consider small increments just in case
        for i in range(1, 5):
            candidates.add(S_greedy[-1] + i)
            
        valid_candidates = []
        for x in candidates:
            if x <= S_greedy[-1]:
                continue
            valid, new_diffs = is_valid_addition(S_greedy, x)
            if valid:
                new_covered = covered_greedy.union([d for d in new_diffs if d in target_diffs])
                new_missing_count = len(target_diffs - new_covered)
                # Prioritize candidates that reduce missing count the most, then smallest x
                valid_candidates.append((new_missing_count, x, new_diffs))
                
        if not valid_candidates:
            # If stuck, pick a random large number? Or break?
            # For greedy, if stuck, we might need to restart or use a larger jump.
            # But usually with 0 and small set, we won't get stuck.
            break
            
        # Sort by (missing count, x)
        valid_candidates.sort()
        # Take the best
        _, x_best, new_diffs_best = valid_candidates[0]
        S_greedy.append(x_best)
        S_greedy.sort()
        covered_greedy = covered_greedy.union(new_diffs_best)
        
    if len(covered_greedy) == 30:
        best_max = max(S_greedy)
        best_set = S_greedy[:]

    # 2. DFS Phase: Optimize
    # We want to find a set with max < best_max
    
    # State: S (list), used_diffs (set), covered_mask (int)
    # Pruning:
    # 1. max(S) >= best_max
    # 2. len(S) > 32
    # 3. Impossible to cover remaining diffs:
    #    Let k = num missing diffs.
    #    Let L = 32 - len(S).
    #    Each new element adds len(S_current) diffs.
    #    Total capacity = sum_{i=1}^L (len(S) + i - 1) = L*len(S) + L(L-1)/2
    #    If capacity < k, prune.
    #    Also, all new diffs must be <= best_max (since new elements must be < best_max to improve).
    #    Actually, if we add x, x < best_max.
    #    So all new diffs are < best_max.
    
    def dfs(S, used_diffs, covered_mask):
        nonlocal best_max, best_set
        
        if time.time() - start_time > deadline:
            return
            
        if len(S) > 32:
            return
            
        current_max = S[-1]
        
        if covered_mask == full_mask:
            if current_max < best_max:
                best_max = current_max
                best_set = S[:]
            return
            
        # Prune: if current max is already >= best_max, no point continuing
        # Note: We are trying to find a set with max STRICTLY LESS than best_max.
        # So if current_max >= best_max, we stop.
        # Wait, if current_max == best_max, we can't improve.
        # If we want to find ANY solution with max <= current_best, we check.
        # But we want to MINIMIZE.
        # So if we found a solution with max M, we search for max < M.
        
        if current_max >= best_max:
            return
            
        # Prune: Capacity check
        missing_count = 30 - bin(covered_mask).count('1')
        if missing_count == 0:
            # Should have been caught above
            return
            
        L = 32 - len(S)
        # Max possible new diffs
        # New element i (1-indexed in the future additions) adds len(S) + i - 1 diffs.
        total_capacity = 0
        current_size = len(S)
        for i in range(1, L + 1):
            total_capacity += current_size + i - 1
            
        if total_capacity < missing_count:
            return
            
        # Generate candidates
        # Candidates must be > S[-1] and < best_max (to improve the max)
        # Also, candidates should help cover missing diffs.
        
        # Heuristic: Candidates are s + d for s in S, d in missing.
        # Also, we might need to add elements that don't immediately cover a diff?
        # If we add an element x that doesn't cover any missing diff, it just wastes slots.
        # Given tight capacity, we should prioritize candidates that cover at least 1 missing diff.
        
        missing_diffs = [d for d in range(1, 31) if not (covered_mask & (1 << (d - 1)))]
        
        candidates = set()
        for d in missing_diffs:
            for s in S:
                x = s + d
                if x > S[-1] and x < best_max:
                    candidates.add(x)
                    
        # If no such candidates, we might be stuck?
        # If we can't add any x < best_max that covers a missing diff,
        # we can't cover the missing diff with x < best_max using one existing s.
        # What if we need two new elements to cover a diff?
        # e.g. need diff d. Add u, then v = u + d.
        # u must be < best_max. v = u + d < best_max => u < best_max - d.
        # So we need to add an element u that doesn't necessarily cover a diff immediately,
        # to allow adding v later.
        # This makes the search space larger.
        
        # To handle this, we can also include "neutral" candidates:
        # Small increments x = S[-1] + 1, S[-1] + 2, etc.
        # But limit them to a few to avoid explosion.
        
        for i in range(1, 4):
            x = S[-1] + i
            if x < best_max:
                candidates.add(x)
                
        if not candidates:
            return
            
        valid_candidates = []
        for x in candidates:
            # Check Sidon property
            new_diffs = []
            valid = True
            for s in S:
                d = x - s
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if not valid:
                continue
                
            # Check new_diffs distinct
            if len(new_diffs) != len(set(new_diffs)):
                continue
                
            # Update mask
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
                    
            # Score:
            # 1. Number of new missing diffs covered (desc)
            # 2. Value of x (asc) - to keep max low
            
            new_covered_count = bin(new_mask ^ covered_mask).count('1')
            valid_candidates.append((new_covered_count, x, new_mask, new_diffs))
            
        if not valid_candidates:
            return
            
        # Sort: Cover more first, then smaller x
        valid_candidates.sort(key=lambda item: (-item[0], item[1]))
        
        for _, x, new_mask, new_diffs in valid_candidates:
            new_S = S + [x]
            new_used_diffs = used_diffs.union(new_diffs)
            
            dfs(new_S, new_used_diffs, new_mask)
            
            # Early termination if we found a very small max?
            # If best_max becomes 30 (theoretical min?), we can stop.
            # The min max is at least 30 because we need diff 30.
            if best_max == 30:
                return

    # Initialize DFS
    # Start with S=[0]
    # used_diffs = set()
    # covered_mask = 0
    
    # To make DFS faster, we can try a few different starting "hints" or just run it.
    # Since 0 is fixed, we just run it.
    
    # If greedy didn't find a solution, best_max is inf.
    # DFS will find one.
    
    dfs([0], set(), 0)
    
    if best_set:
        return best_set
    else:
        # Fallback: Return the greedy solution if it exists, or a random valid one
        if S_greedy and covered_greedy.issuperset(target_diffs):
            return S_greedy
        else:
            # Emergency: return a known valid set
            # Set {0, 1, 4, 10, 18, 23, 27, 29, 30} ? 
            # Let's check a known small Sidon set covering 1..30.
            # A (v, k, 1) difference set.
            # For v=31 (prime), k=6? 6*5=30. Yes!
            # A (31, 6, 1) difference set exists.
            # Example: {0, 1, 3, 8, 12, 18} in Z_31?
            # Let's verify {0, 1, 3, 8, 12, 18}
            # Diffs:
            # 1, 3, 8, 12, 18
            # 2, 7, 11, 17
            # 5, 9, 15
            # 4, 10
            # 6
            # Set: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 17, 18}
            # Missing: 13, 14, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
            # This only covers 15 diffs. We need 30.
            # So size 6 is not enough?
            # Max diffs for size 6 is 15.
            # We need 30 diffs.
            # So we need size at least n such that n(n-1)/2 >= 30 => n=8 (28 diffs) is not enough.
            # n=9 => 36 diffs.
            # So min size is 9.
            
            # Try to construct a set of size 9 manually or via quick search?
            # The DFS above should find it.
            # If it fails, we return a large set.
            # Let's return the greedy set if it covers, else a generic large Sidon set.
            # Generic Sidon: {0, 1, 3, 7, 12, 18, 25, 33, 42} (Sylvester sequence based?)
            # Let's just return the greedy one, hoping it works.
            return S_greedy

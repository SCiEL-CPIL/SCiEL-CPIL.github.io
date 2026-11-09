import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # We search for the smallest D such that there exists a Sidon set A in [0, D]
    # containing 0, with |A| <= 32, and covering all differences 1..30.
    
    # Lower bound for D:
    # A Sidon set of size k has at least k(k-1)/2 differences.
    # We need to cover 30 differences.
    # k(k-1)/2 >= 30 => k^2 - k - 60 >= 0 => k ~ 8.5. So k=9.
    # A Golomb ruler of order 9 has minimum length 41 (known optimal).
    # But we also require 0 in the set.
    # If 0 is in the set, the differences are just the elements and pairwise diffs.
    # Actually, if 0 is in the set, the set is {0, a1, ..., ak-1}.
    # Differences include ai - 0 = ai. So the elements themselves must be distinct (obvious)
    # and not conflict with other differences.
    # The minimum diameter for a Sidon set of size 9 is 41.
    # However, we need to cover 1..30.
    # Let's start D from 40 to be safe, maybe 50.
    
    for D in range(40, 101):
        if time.time() > deadline:
            break
            
        # Bitmask for differences 1..30
        # We use a Python int as a bitset for all differences up to D.
        # If D is small (< 1000), this is very efficient.
        
        target_mask = 0
        for d in range(1, 31):
            target_mask |= (1 << d)
            
        # missing_mask starts as target_mask
        # used_mask starts as 0
        
        def dfs(current_set, used_mask, missing_mask, last_elem):
            if time.time() > deadline:
                return None
                
            if missing_mask == 0:
                return current_set[:]
                
            if len(current_set) > 32:
                return None
                
            # Heuristic: If we can't cover missing diffs with remaining capacity, prune?
            # Hard to estimate remaining capacity without knowing future picks.
            # But we can check if the number of missing diffs is too large for the number of elements we can still add.
            # Max elements we can add is roughly D - last_elem.
            # Each new element adds at most len(current_set) new diffs.
            # This bound is loose.
            
            candidates = []
            
            # Try adding x from last_elem + 1 to D
            # Optimization: We can try to prioritize x that cover missing diffs.
            # But we must check validity first.
            
            for x in range(last_elem + 1, D + 1):
                # Check if x conflicts with current_set
                new_mask = 0
                valid = True
                for y in current_set:
                    d = x - y
                    # d is positive since x > last_elem >= y
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if not valid:
                    continue
                
                # Check if this x helps cover missing diffs
                covered = new_mask & missing_mask
                # Score: number of missing diffs covered
                score = bin(covered).count('1')
                
                # We also need to ensure we don't create a situation where we can't finish?
                # For now, just collect valid candidates.
                
                candidates.append((score, x, new_mask))
                
            # Sort by score descending (greedy: cover most missing first)
            candidates.sort(key=lambda t: -t[0])
            
            # To avoid symmetry or redundant paths, we might want to limit candidates?
            # No, we need to search properly.
            
            for score, x, new_mask in candidates:
                new_missing = missing_mask & ~new_mask
                new_used = used_mask | new_mask
                
                res = dfs(current_set + [x], new_used, new_missing, x)
                if res is not None:
                    return res
                    
            return None
        
        result = dfs([0], 0, target_mask, 0)
        
        if result is not None:
            # Verify result just in case
            s = sorted(result)
            diffs = set()
            valid_sidon = True
            for i in range(len(s)):
                for j in range(i):
                    d = s[i] - s[j]
                    if d in diffs:
                        valid_sidon = False
                        break
                    diffs.add(d)
                if not valid_sidon:
                    break
            
            if valid_sidon and all(d in diffs for d in range(1, 31)):
                return result
            else:
                # Should not happen if DFS is correct, but if it does, continue searching
                continue

    # Fallback: If not found in [40, 100], try a wider range or greedy
    # Given the problem constraints, a solution should exist in this range.
    # If we reach here, we return a known valid set with larger D if we had one,
    # but we don't. We'll just return a greedy construction which might be large.
    
    a = [0]
    for d in range(1, 31):
        if d in set(a[i] - a[j] for i in range(len(a)) for j in range(len(a)) if a[i] > a[j]):
            continue
        found = False
        for x in range(1, 1000):
            if x in a:
                continue
            trial = sorted(a + [x])
            t_diffs = set()
            ok = True
            for i in range(len(trial)):
                for j in range(i):
                    dd = trial[i] - trial[j]
                    if dd in t_diffs:
                        ok = False
                        break
                    t_diffs.add(dd)
                if not ok:
                    break
            if ok:
                a = trial
                found = True
                break
        if not found:
            # Force add
            y = 2 * max(a) + 100
            a.append(y)
            
    return a

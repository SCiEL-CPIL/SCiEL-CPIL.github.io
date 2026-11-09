"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # We need a Sidon set S with 0 in S, all differences 1..30 present,
    # minimize max(S). Use backtracking.
    
    # Bitmask for differences 1..60 (we'll track which differences are used)
    # Actually, differences can be larger, but we only care about 1..30 being covered
    # and all differences being distinct.
    
    best = None
    best_max = 10**6
    
    def search(current_set, used_diffs, covered, max_allowed):
        nonlocal best, best_max
        if time.time() > deadline:
            return
        
        # Pruning: if current max already >= best_max, stop
        if current_set and max(current_set) >= best_max:
            return
        
        # Check if we've covered all 1..30
        if all(covered[i] for i in range(1, 31)):
            m = max(current_set)
            if m < best_max:
                best_max = m
                best = list(current_set)
            return
        
        # Count how many of 1..30 are still uncovered
        uncovered = [d for d in range(1, 31) if not covered[d]]
        if not uncovered:
            m = max(current_set)
            if m < best_max:
                best_max = m
                best = list(current_set)
            return
        
        # If we have too many elements, stop
        if len(current_set) >= 32:
            return
        
        # Pruning: with remaining elements, can we cover all uncovered diffs?
        # Each new element adds at most len(current_set) new differences
        remaining_slots = 32 - len(current_set)
        # Max new differences we can add
        max_new_diffs = 0
        for i in range(remaining_slots):
            max_new_diffs += len(current_set) + i
        if max_new_diffs < len(uncovered):
            return
        
        # Try adding new elements
        # The new element x must be > 0 (since 0 is in set and we want positive diffs)
        # x - a for each a in current_set must be positive and not in used_diffs
        # Also x - a must be such that for all a, b in current_set, x-a != b-a' unless a==a' and b==b'
        # Actually, the Sidon condition: all pairwise differences are distinct.
        # When adding x, new differences are x - a for a in current_set (all positive since x > all in current_set if we keep sorted)
        # These must not conflict with existing differences.
        
        # What range of x to try?
        # x must be at least min(uncovered) + min(current_set) = min(uncovered) (since 0 in set)
        # Actually x - 0 = x, so x must be in uncovered or x must create other useful diffs
        # But x could be larger than 30
        
        # Let's try x from 1 to max_allowed
        # But we can prune: x must be such that x - a is not in used_diffs for any a in current_set
        # And x - a > 0 for all a in current_set (so x > max(current_set) if we keep sorted)
        
        # Actually, let's keep the set sorted and always add x > max(current_set)
        # This ensures all new differences x - a are positive
        
        min_x = max(current_set) + 1
        # Upper bound: we want to keep max small
        # x should be at most best_max - 1 to be useful
        # Also, x - 0 = x, so if x > 30, it doesn't directly help cover 1..30
        # But x - a for a > 0 could be in 1..30
        
        # Let's bound x: x <= best_max - 1
        # And x - min(current_set) should be reasonable
        max_x = min(best_max - 1, max_allowed)
        
        if min_x > max_x:
            return
        
        # For each candidate x, check if it's valid
        for x in range(min_x, max_x + 1):
            if time.time() > deadline:
                return
            
            # Check new differences
            new_diffs = []
            valid = True
            for a in current_set:
                d = x - a
                if d <= 0:
                    valid = False
                    break
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Check that new_diffs are all distinct among themselves
            if len(set(new_diffs)) != len(new_diffs):
                continue
            
            # This is valid! Recurse
            new_set = current_set + [x]
            new_used = used_diffs | set(new_diffs)
            new_covered = list(covered)
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_covered[d] = True
            
            search(new_set, new_used, new_covered, max_allowed)
    
    # Initial state: {0}, no used diffs, no covered diffs
    # covered is a list indexed 1..30
    covered = [False] * 31
    used_diffs = set()
    
    # Start with just {0}
    search([0], used_diffs, covered, 10**6)
    
    if best is not None:
        return best
    
    # Fallback: greedy approach
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

# EVOLVE-BLOCK-END

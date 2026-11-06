import time
import sys

def construct():
    start_time = time.time()
    timeout = 140.0  # 10s margin before 150s kill

    target_diffs = set(range(1, 31))
    
    # We will use DFS
    # State: A (sorted list), diffs (set)
    
    def is_valid_addition(A, diffs, x):
        # x must be > max(A)
        if x <= max(A):
            return False
        new_diffs = []
        for a in A:
            d = x - a
            if d in diffs:
                return False
            new_diffs.append(d)
        return True

    def count_target_coverage(A, diffs, x):
        # Returns number of new differences that are in target_diffs and not in diffs
        count = 0
        for a in A:
            d = x - a
            if d in target_diffs and d not in diffs:
                count += 1
        return count

    def dfs(A, diffs):
        if time.time() - start_time > timeout:
            return None
            
        missing = target_diffs - diffs
        if not missing:
            return A[:]
            
        # If we have too many elements, stop? 
        # Max 32 elements.
        if len(A) >= 32:
            return None
            
        # Generate candidates
        # We want to try x that cover many missing diffs
        # Also, x should be small to keep max(A) small
        # But x must be > max(A)
        
        max_a = max(A)
        
        # We can bound x. 
        # If x is very large, the differences x-a will be large.
        # We only care about differences in 1..30 for coverage.
        # But we also need to avoid collisions.
        
        # Let's try x in a range.
        # A reasonable upper bound for x in the next step?
        # If we are at step k, we need to cover remaining diffs.
        # Let's just iterate x from max_a + 1 to some limit.
        # What limit? 
        # If we pick x > max_a + 30, then x - max_a > 30.
        # The smallest new diff is x - max_a.
        # If x - max_a > 30, then NO new difference can cover any missing target diff (since all targets <= 30).
        # So if we pick x such that x - max_a > 30, we make no progress on coverage.
        # This might be a dead end unless we *must* add an element for some other reason? 
        # No, we only add elements to cover diffs.
        # So we should restrict x such that x - max_a <= 30?
        # Wait. If A={0, 100}, missing={1}. 
        # If we pick x=101, diff=1. Covered.
        # If we pick x=150, diff=50, 100. No progress.
        # So yes, we should only consider x where at least one new difference is in missing.
        # This implies x - a <= 30 for some a in A.
        # Since a <= max_a, x - max_a <= 30 is a necessary condition for x to cover *any* missing diff?
        # Not exactly. If A={0, 10, 20}, missing={1}.
        # x=21 -> diff 1 (21-20), 11, 21. Covers 1.
        # x=31 -> diff 31, 21, 11. No cover.
        # So x must be close to some a.
        # Specifically, x must be in the union of intervals [a+1, a+30] for a in A?
        # And x > max_a.
        
        candidates = []
        # x must be > max_a
        # And x - a must be in missing for at least one a?
        # Actually, we can just iterate x from max_a + 1 to max_a + 30.
        # Why 30? Because if x - max_a > 30, then for all a in A, x - a >= x - max_a > 30.
        # So no new difference is <= 30.
        # Thus no new target diff is covered.
        # So such an x is useless for covering missing targets.
        # We can skip it.
        
        for x in range(max_a + 1, max_a + 31):
            if not is_valid_addition(A, diffs, x):
                continue
            cov = count_target_coverage(A, diffs, x)
            if cov > 0:
                candidates.append((cov, x))
                
        # If no candidates with cov > 0, we might be stuck?
        # Or we need to add an element that doesn't cover targets immediately but enables future coverage?
        # No, adding an element that doesn't cover targets just adds "noise" differences.
        # It might block future coverage.
        # So if no candidate covers a target, we should probably backtrack.
        
        if not candidates:
            return None
            
        # Sort by coverage (desc), then by x (asc) to keep values small
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for cov, x in candidates:
            new_A = A + [x]
            new_diffs = set(diffs)
            for a in A:
                new_diffs.add(x - a)
                
            res = dfs(new_A, new_diffs)
            if res is not None:
                return res
                
        return None

    # Initial state
    A = [0]
    diffs = set()
    
    res = dfs(A, diffs)
    
    if res is not None:
        return res
    else:
        # Fallback: return a valid set if found by other means, or empty?
        # The task requires a valid set.
        # If DFS fails due to timeout or dead end, we might need a greedy fallback.
        # But the DFS with this heuristic should find a solution.
        # Let's add a simple greedy fallback just in case.
        a = [0]
        def diffs_list(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs_list(sorted(a)):
                continue
            found = False
            for x in range(max(a) + 1, max(a) + 100):
                trial = sorted(a + [x])
                dl = diffs_list(trial)
                if len(set(dl)) == len(dl):
                    a = trial
                    found = True
                    break
            if not found:
                # Try larger
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y])
                    dl = diffs_list(trial)
                    if len(set(dl)) == len(dl):
                        a = trial
                        break
                    y += 1
        return a

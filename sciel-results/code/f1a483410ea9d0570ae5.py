"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    # Target: find a Sidon set containing 0 with all differences 1..30
    # Minimize the largest element (diameter)
    
    # A Sidon set (B2 sequence) has the property that all positive differences are distinct.
    # We need differences 1, 2, ..., 30 to all appear.
    
    # Strategy: 
    # 1. Try small diameters M (starting from a lower bound)
    # 2. For each M, search for a valid Sidon set in [0, M] containing 0
    #    that covers all differences 1..30
    
    # Lower bound on M: 
    # If we have n elements, max differences is n*(n-1)/2.
    # We need at least 30 differences, so n*(n-1)/2 >= 30 => n >= 9 (since 8*7/2=28, 9*8/2=36)
    # But we also need specific small differences. 
    # Known small Sidon sets: {0, 1, 4, 10, 12, 17, 21, 23, 24} has diameter 24 but may not cover 1..30.
    # Let's try increasing M.
    
    # Precompute: for a given set, check if it's Sidon and covers 1..30
    def is_valid_solution(s):
        """Check if s is a valid Sidon set with all differences 1..30."""
        s = sorted(s)
        if 0 not in s:
            return False
        diffs = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.add(d)
        # Check all 1..30 are present
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    def count_missing_diffs(s, needed):
        """Count how many of the needed differences are missing."""
        s = sorted(s)
        diffs = set()
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                diffs.add(s[j] - s[i])
        return len(needed - diffs)
    
    # DFS with pruning for a given diameter M
    def dfs(M, current_set, used_diffs, depth, max_depth):
        """
        Search for a Sidon set in [0, M] containing 0.
        current_set: sorted list of elements chosen so far
        used_diffs: set of differences already used
        depth: current depth
        max_depth: maximum number of elements to try
        Returns: list if found, None otherwise
        """
        if time.time() > DEADLINE:
            return None
        
        # Check if we've covered all required differences
        if all(d in used_diffs for d in range(1, 31)):
            return list(current_set)
        
        if depth >= max_depth:
            return None
        
        # If we can't possibly cover all remaining differences with remaining slots
        remaining_needed = [d for d in range(1, 31) if d not in used_diffs]
        if not remaining_needed:
            return list(current_set)
        
        # Pruning: if the number of remaining elements we can add is too small
        # to generate enough new differences, prune
        # Each new element can create at most len(current_set) new differences
        # Actually, adding one element x to current_set creates len(current_set) new differences
        # So with (max_depth - depth) more elements, we can create at most 
        # sum of len(current_set) + i for i in range(max_depth - depth) new diffs
        # This is a rough upper bound
        
        # Better pruning: if the smallest missing difference is larger than M, we're stuck
        # (but all our needed diffs are <= 30 and M >= 30, so this is fine)
        
        # Try adding new elements
        last = current_set[-1] if current_set else -1
        start_candidate = last + 1 if last >= 0 else 0
        
        for x in range(start_candidate, M + 1):
            if time.time() > DEADLINE:
                return None
            
            # Check if adding x creates duplicate differences
            new_diffs = []
            conflict = False
            for y in current_set:
                d = x - y
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            # Check if new_diffs have duplicates among themselves
            # (they can't since y are all different and x is fixed, so d values are distinct)
            
            # Recurse
            new_used = used_diffs.union(new_diffs)
            result = dfs(M, current_set + [x], new_used, depth + 1, max_depth)
            if result is not None:
                return result
        
        return None
    
    # Try increasing diameters
    # Lower bound: need to cover diff 30, so M >= 30
    # But Sidon constraints require larger M.
    # Known: smallest Sidon set with 1..30 might have diameter around 40-60
    
    # Start from a reasonable lower bound
    # For n elements, the minimum diameter for a Sidon set covering 1..k is roughly sqrt(k*n)
    # Let's try M from 30 upward
    
    best_solution = None
    
    # First, try a greedy approach to get an initial solution
    def greedy_construct():
        """Greedy: add elements to cover missing differences."""
        a = [0]
        used_diffs = set()
        
        for d in range(1, 31):
            if d in used_diffs:
                continue
            # Try to find an x such that adding x to a creates difference d
            # x - y = d for some y in a, or y - x = d for some y in a
            found = False
            for y in a:
                # x = y + d
                x = y + d
                if x < 0:
                    continue
                # Check if adding x is valid
                new_diffs = set()
                conflict = False
                for z in a:
                    dd = abs(x - z)
                    if dd == 0:
                        continue
                    if dd in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(dd)
                if not conflict:
                    a.append(x)
                    a.sort()
                    used_diffs.update(new_diffs)
                    found = True
                    break
                # x = y - d
                x = y - d
                if x < 0:
                    continue
                new_diffs = set()
                conflict = False
                for z in a:
                    dd = abs(x - z)
                    if dd == 0:
                        continue
                    if dd in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(dd)
                if not conflict:
                    a.append(x)
                    a.sort()
                    used_diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                # Add a new element far away
                x = 2 * max(a) + d + 1
                while True:
                    new_diffs = set()
                    conflict = False
                    for z in a:
                        dd = x - z
                        if dd in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(dd)
                    if not conflict:
                        a.append(x)
                        a.sort()
                        used_diffs.update(new_diffs)
                        break
                    x += 1
        return a
    
    # Get a greedy solution first
    greedy_sol = greedy_construct()
    if is_valid_solution(greedy_sol):
        best_solution = greedy_sol
        best_diameter = max(greedy_sol)
    else:
        best_diameter = 1000000  # Large number
    
    # Now try to improve with DFS for smaller diameters
    # Start from a lower bound and go up
    # Lower bound: at least 30 (to have difference 30)
    # But Sidon constraints: for 9 elements, min diameter is around 30-40
    
    # Try diameters from 30 to best_diameter - 1
    for M in range(30, best_diameter):
        if time.time() > DEADLINE:
            break
        
        # Try different max sizes for the set
        # Minimum size: 9 (since 8*7/2 = 28 < 30)
        # Maximum size: we can have up to ~10-12 elements
        for max_depth in range(9, 15):
            if time.time() > DEADLINE:
                break
            
            result = dfs(M, [0], set(), 1, max_depth)
            if result is not None:
                if is_valid_solution(result):
                    best_solution = result
                    best_diameter = M
                    break
        
        if best_solution is not None and max(best_solution) < M:
            # We found a solution with smaller diameter
            break
    
    if best_solution is None:
        # Fallback to greedy
        best_solution = greedy_sol
    
    return best_solution
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # We need a Sidon set (B_2 set) containing 0, with all differences 1..30 covered.
    # Minimize the largest element (diameter).
    
    # A Sidon set of size n has diameter at least ~n^2/4 roughly, but we only need 32 elements max.
    # Actually, the minimal diameter for a Sidon set covering 1..30 is what we seek.
    # 
    # Strategy: DFS/backtracking search for a Sidon set that covers all differences 1..30.
    # We build the set incrementally, ensuring Sidon property at each step.
    # We try to keep elements small.
    
    # First, let's try a direct DFS approach:
    # - Start with {0}
    # - At each step, try adding a new element x (in increasing order from 1)
    # - Check that all pairwise differences are distinct (Sidon property)
    # - Track which differences 1..30 have been covered
    # - Stop when all 30 are covered
    # - Minimize the maximum element
    
    # For efficiency, we use a bitmask for covered differences 1..30.
    # We also maintain the set of all differences to check for Sidon property.
    
    best_solution = None
    best_max_val = float('inf')
    
    # Use a more targeted approach:
    # The minimum possible diameter for a Sidon set with 32 elements covering diffs 1-30...
    # A perfect difference set of order 5 (size 11, diameter 31) covers 0..30? No, that's different.
    # Let's just do a smart search.
    
    # Approach: Iterative deepening DFS with pruning.
    # Try max_val from small to large, and for each, do a DFS to find a valid Sidon set.
    
    # Actually, let's do a branch-and-bound DFS:
    # - Build set incrementally
    # - Current max element is an upper bound on diameter
    # - If current max >= best_max_val, prune
    
    # For speed, we'll use a heuristic: try small values first, and use the fact that
    # we need to cover specific differences.
    
    # Let's use a different strategy: 
    # 1. First try known constructions
    # 2. Then do a randomized/local search
    
    # Known result: The minimal diameter for a Sidon set covering 1..30 with 0 included
    # is related to perfect difference sets. A (11, 5, 1) difference set has size 11 and 
    # covers differences 1..30 exactly once (cyclic). But we need a Sidon set in integers.
    
    # Let's try a DFS with good pruning:
    
    def solve_dfs():
        nonlocal best_solution, best_max_val, deadline
        
        # We'll search for the minimum diameter
        # Start with a reasonable lower bound and increase
        
        # Lower bound: we need at least enough elements to cover 30 differences.
        # A set of size k has k*(k-1)/2 differences. Need >= 30, so k >= 8.
        # But we also need 0 in the set, and the Sidon constraint.
        
        # Let's do iterative deepening on the maximum value M.
        # For each M, check if there's a valid Sidon subset of {0..M} containing 0
        # that covers all differences 1..30.
        
        # But this is exponential. Instead, let's do a direct DFS building the set.
        
        # State: current set (sorted), set of all diffs, bitmask of covered diffs 1-30
        # We add elements one at a time, trying small values first.
        
        # For efficiency, limit the number of elements to at most 32.
        
        # Key insight: in a Sidon set, all pairwise differences are distinct.
        # So if we have elements a1 < a2 < ... < ak, then all (ai - aj) for i > j are distinct.
        
        # Let's use a recursive search with memoization-like pruning.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(current_set, all_diffs, covered_mask, depth):
            nonlocal best_solution, best_max_val, deadline
            
            if time.time() > deadline:
                return False
            
            # Check if we've covered all differences 1..30
            if covered_mask == (1 << 30) - 1:
                max_val = current_set[-1] if current_set else 0
                if max_val < best_max_val:
                    best_max_val = max_val
                    best_solution = list(current_set)
                return True
            
            # Pruning: if we have too many elements
            if len(current_set) >= 32:
                return False
            
            # Pruning: if current max already >= best_max_val
            if current_set and current_set[-1] >= best_max_val:
                return False
            
            # Determine which differences are still needed
            remaining = [(1 << 30) - 1] ^ covered_mask
            if remaining == 0:
                return True
            
            # Try adding new elements
            # Start from 1 up to some limit
            # We need to be smart about what values to try
            
            # The next element x must be > 0 (since 0 is already in the set)
            # and x - a for all a in current_set must not be in all_diffs (Sidon property)
            # and x - a for all a in current_set must be positive (since x > all a if we add in order)
            # Wait, we don't have to add in sorted order necessarily, but it's convenient.
            
            # Actually, let's always maintain the set in sorted order and only add elements
            # larger than the current max. This ensures we build up from 0.
            
            # But this might miss solutions where a large element is needed early.
            # However, since we're minimizing the max, adding in increasing order is fine
            # because any solution with max M can be reordered.
            
            # Hmm, but the Sidon property depends on all pairwise differences, not just
            # differences from the max. So we need to check all pairs.
            
            # Let's try values from 1 to min(best_max_val - 1, some_upper_bound)
            # But we also need to cover the remaining differences.
            
            # For each remaining difference d, we need x - a = d for some a in current_set,
            # or a - x = d (but x > a always since we add in order, so only x - a = d).
            # Wait, no: if we add x > all current elements, then new differences are x - a for all a.
            # These must all be new (not in all_diffs) and we get coverage of those differences.
            
            # But we might also need differences that can only be formed as a - x where x < a,
            # but since we're adding x > max(current_set), this can't happen.
            # So we can only cover differences d where d = x - a for some a in current_set.
            
            # This means: for a given x, the new differences are {x - a : a in current_set}.
            # We need these to be all distinct from each other (which they are, since a's are distinct)
            # and distinct from all existing differences in all_diffs.
            
            # So the check is: for all a1 != a2 in current_set, (x - a1) != (x - a2) [always true]
            # and for all a in current_set, (x - a) not in all_diffs.
            
            # Let's implement this.
            
            # What range of x to try?
            # x must be at least 1 (since 0 is in the set)
            # x must be less than best_max_val (for pruning)
            # Also, x should be such that x - a covers some remaining difference for some a.
            
            # For efficiency, let's generate candidate x values:
            # For each remaining difference d and each a in current_set, x = a + d is a candidate.
            # This ensures that adding x covers difference d.
            
            candidates = set()
            for d_bit in range(30):
                if remaining & (1 << d_bit):
                    d = d_bit + 1
                    for a in current_set:
                        x = a + d
                        if 1 <= x < best_max_val:
                            candidates.add(x)
            
            # Also try some small values that might not directly cover a difference
            # but might be useful for future steps. Actually, in a Sidon set, every element
            # contributes to covering differences, so we should focus on candidates.
            
            # Sort candidates for deterministic behavior and to try smaller values first
            for x in sorted(candidates):
                if time.time() > deadline:
                    return False
                
                # Check Sidon property: x - a not in all_diffs for all a in current_set
                valid = True
                new_diffs = []
                for a in current_set:
                    diff = x - a
                    if diff in all_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if not valid:
                    continue
                
                # Check that new_diffs are all distinct (they should be since a's are distinct)
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                # Update state
                new_all_diffs = set(all_diffs)
                new_covered = covered_mask
                for diff in new_diffs:
                    new_all_diffs.add(diff)
                    if 1 <= diff <= 30:
                        new_covered |= (1 << (diff - 1))
                
                new_set = current_set + [x]
                
                if dfs(new_set, new_all_diffs, new_covered, depth + 1):
                    return True
            
            return False
        
        # Initial state
        initial_set = [0]
        initial_diffs = set()
        initial_mask = 0
        
        dfs(initial_set, initial_diffs, initial_mask, 0)
    
    # First, try a direct DFS with a reasonable upper bound
    # Start with a loose bound and tighten
    best_max_val = 10**6
    
    # Try with a tighter initial bound based on known results
    # A (v, k, lambda) difference set with v=31, k=11, lambda=1 gives a cyclic difference set.
    # The corresponding Sidon set in integers would have diameter around 30-40.
    # Let's try with an upper bound of 100 first.
    
    best_max_val = 100
    solve_dfs()
    
    if best_solution is not None:
        return best_solution
    
    # If DFS didn't find a solution with max < 100, try with a larger bound
    best_max_val = 200
    solve_dfs()
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: use the original greedy approach
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

# EVOLVE-BLOCK-END

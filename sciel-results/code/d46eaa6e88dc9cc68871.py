"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # We need a Sidon set containing 0, with all positive differences distinct,
    # and every d in 1..30 occurring as a difference.
    # Minimize the largest element.
    
    # Strategy: Backtracking search.
    # We build the set incrementally.
    # Key insight: The set must contain 0.
    # Let S be the set. 0 in S.
    # Differences are |a - b| for a, b in S, a != b.
    # All differences must be distinct (Sidon property for differences).
    # We need differences 1, 2, ..., 30 all present.
    
    # A Sidon set of size n has at most n(n-1)/2 differences.
    # We need at least 30 differences. So n(n-1)/2 >= 30 => n >= 8 (8*7/2=28 < 30, 9*8/2=36 >= 30).
    # So we need at least 9 elements.
    # The problem says "at most 32 distinct integers".
    
    # Let's use a backtracking search that builds the set element by element.
    # To minimize the largest element, we should search in order of increasing max element.
    # But that's hard to do directly. Instead, we can do a DFS that tries to keep elements small.
    
    # Better approach: 
    # 1. Precompute which pairs of differences are "compatible".
    #    Two differences d1, d2 are incompatible if they can't both be in a Sidon set with 0.
    #    Actually, the Sidon constraint is on the set elements, not just differences.
    #    If we have elements {0, a, b, ...}, then |a-b| must not equal |a-0|, |b-0|, or any other difference.
    
    # Let's think about this differently.
    # We need to find a set S = {0, x1, x2, ..., xk} such that:
    # - All elements are distinct, non-negative integers
    # - All positive differences |xi - xj| are distinct
    # - The set of differences contains {1, 2, ..., 30}
    # - Minimize max(S)
    
    # This is equivalent to finding a Golomb ruler of length max(S) with marks at positions in S,
    # such that all differences 1..30 are represented.
    # A Golomb ruler is a Sidon set.
    
    # Known result: The optimal Golomb ruler that covers all differences 1..n is related to perfect difference sets.
    # For n=30, we can search.
    
    # Approach: Backtracking with constraint propagation.
    # We maintain:
    # - current set S (sorted)
    # - set of used differences
    # - set of required differences not yet covered
    # - next element to try
    
    # To keep the search tractable, we use the following heuristics:
    # 1. Always try to cover the smallest uncovered required difference first.
    # 2. When adding a new element x, check that it doesn't create duplicate differences.
    
    # Let's implement a DFS.
    
    best_result = None
    best_max_val = float('inf')
    
    # We'll search for solutions with max element up to some bound.
    # Start with a reasonable bound and increase if needed.
    # From literature, the minimal Golomb ruler covering 1..30 has max element around 100-200.
    
    # Actually, let's just do a DFS that tries to minimize the max element.
    # We'll use branch and bound: if current max element >= best_max_val, prune.
    
    def dfs(S, used_diffs, required_remaining, max_val_so_far):
        nonlocal best_result, best_max_val
        
        if time.time() > deadline:
            return
        
        # Check if we've covered all required differences
        if not required_remaining:
            if max_val_so_far < best_max_val:
                best_max_val = max_val_so_far
                best_result = list(S)
            return
        
        # Branch and bound
        if max_val_so_far >= best_max_val:
            return
        
        # Choose the smallest uncovered required difference to cover next
        target_diff = min(required_remaining)
        
        # We need to add one or more elements to cover target_diff.
        # The new element x must satisfy: x - y = target_diff or y - x = target_diff for some y in S.
        # Since S contains 0, x = target_diff or x = -target_diff (but x >= 0, so x = target_diff if 0 in S).
        # More generally, for each y in S, x = y + target_diff or x = y - target_diff (if >= 0).
        
        # Generate candidate x values
        candidates = set()
        for y in S:
            candidates.add(y + target_diff)
            if y - target_diff >= 0:
                candidates.add(y - target_diff)
        
        # Sort candidates to try smaller ones first (to minimize max element)
        candidates = sorted(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                return
            
            # Check if x is already in S
            if x in S:
                continue
            
            # Check if adding x creates duplicate differences
            new_S = sorted(S | {x})
            new_diffs = set()
            valid = True
            for i in range(len(new_S)):
                for j in range(i+1, len(new_S)):
                    d = new_S[j] - new_S[i]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if x is too large
            new_max = max(new_S)
            if new_max >= best_max_val:
                continue
            
            # Update used differences and required remaining
            new_used_diffs = used_diffs | new_diffs
            new_required = required_remaining - new_diffs
            
            # Recurse
            dfs(frozenset(new_S), new_used_diffs, new_required, new_max)
    
    # Initial state
    initial_S = frozenset([0])
    initial_used_diffs = set()
    initial_required = set(range(1, 31))
    
    dfs(initial_S, initial_used_diffs, initial_required, 0)
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: use the greedy approach from the original program
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

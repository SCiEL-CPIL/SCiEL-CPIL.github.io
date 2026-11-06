import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s limit

    n = 256
    best_A = []
    best_B = []
    best_size = 0

    def get_sums(S):
        return {a + b for a, b in combinations_with_replacement(S, 2)}

    def get_diffs(S):
        return {b - a for a, b in combinations(S, 2)}

    def is_valid_state(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        sums_A = get_sums(A)
        if len(sums_A) != len(list(combinations_with_replacement(A, 2))):
            return False
        sums_B = get_sums(B)
        if len(sums_B) != len(list(combinations_with_replacement(B, 2))):
            return False
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        if diffs_A & diffs_B:
            return False
        return True

    # Build conflict graph:
    # Vertices: (x, 'A') and (x, 'B') for x in 1..n
    # Edge between (x, s1) and (y, s2) if placing x in set s1 and y in set s2 violates constraints
    # We model this as: for each pair (x, y) with x != y, determine which (set1, set2) assignments conflict.
    # Also self-conflicts: (x, s) vs (x, s) is fine (same element same set), but (x, A) vs (x, B) 
    # means x is in both A and B - that's allowed per problem statement.
    
    # Let's define:
    # conflict_A[x][y] = True if x and y cannot both be in A (Sidon violation)
    # conflict_B[x][y] = True if x and y cannot both be in B (Sidon violation)  
    # conflict_AB[x][y] = True if x in A and y in B violates diff disjointness
    # conflict_BA[x][y] = True if y in A and x in B violates diff disjointness (symmetric with above if we define properly)
    
    # Actually let's think more carefully:
    # Sidon for A: for any distinct a1,a2,a3,a4 in A, a1+a2 != a3+a4 (with multiplicity for pairs)
    # This is equivalent to: the multiset of pair sums {a_i + a_j : i <= j} has all unique values.
    # So x,y conflict in A if there exist a,b in A\{x,y} such that x+y = a+b, or x+x = a+b, etc.
    # But for the conflict graph approach, we need: given current partial assignment, can we add (x, 'A')?
    
    # Let me use a different approach: precompute for each pair (x,y) whether they can coexist in the same Sidon set.
    # Two elements x,y can be in the same Sidon set iff 2x != 2y (trivially true if x!=y) and 
    # the key issue is: if x,y,z,w are all in the set, we need x+y != z+w for all distinct pairs.
    # For just two elements x,y in a set, there's no conflict (a 2-element set is always Sidon).
    # The conflict arises when we have 3+ elements.
    
    # Better approach: use the incremental check with precomputed sum/diff sets, but make it faster
    # by using bitsets or by precomputing which pairs of elements create sum collisions.
    
    # Let me precompute: for each x, the set of y values such that x and y would create a sum collision
    # with some other pair. This is complex. Let me instead use a smarter greedy with better ordering.
    
    # Alternative: Precompute all "forbidden pairs" for Sidon:
    # (x, y) is forbidden in set S if there exist (a, b) with a <= b, a != x, b != y (or a != y, b != x)
    # such that x + y = a + b. But we don't know a, b yet.
    
    # Let me use the incremental approach but with NumPy for speed.
    
    import numpy as np

    def fast_greedy(seed_offset=0):
        """Fast greedy construction using precomputed conflict checks."""
        rng = random.Random(seed_offset)
        
        A = set()
        B = set()
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        # Precompute: for each x, what sums would x create with each existing element
        # Use a different strategy: try elements in randomized order, but be smarter about which set to assign to
        
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Try adding to A first (or randomly choose which to try first)
            try_A = rng.random() < 0.5
            
            if try_A:
                # Try A then B
                for target in ['A', 'B']:
                    if target == 'A':
                        if len(A) >= 64:
                            continue
                        # Check Sidon: new sums = {x+s for s in A} U {2x}
                        new_sums = {x + s for s in A}
                        new_sums.add(2 * x)
                        if new_sums & sums_A:
                            continue
                        # Check diff disjointness: new diffs = {x-s, s-x for s in A}
                        new_diffs = set()
                        for s in A:
                            new_diffs.add(x - s)
                            new_diffs.add(s - x)
                        if new_diffs & diffs_B:
                            continue
                        # Valid - add to A
                        A.add(x)
                        sums_A.update(new_sums)
                        diffs_A.update(new_diffs)
                        break
                    else:  # target == 'B'
                        if len(B) >= 64:
                            continue
                        new_sums = {x + s for s in B}
                        new_sums.add(2 * x)
                        if new_sums & sums_B:
                            continue
                        new_diffs = set()
                        for s in B:
                            new_diffs.add(x - s)
                            new_diffs.add(s - x)
                        if new_diffs & diffs_A:
                            continue
                        B.add(x)
                        sums_B.update(new_sums)
                        diffs_B.update(new_diffs)
                        break
            else:
                # Try B then A
                for target in ['B', 'A']:
                    if target == 'B':
                        if len(B) >= 64:
                            continue
                        new_sums = {x + s for s in B}
                        new_sums.add(2 * x)
                        if new_sums & sums_B:
                            continue
                        new_diffs = set()
                        for s in B:
                            new_diffs.add(x - s)
                            new_diffs.add(s - x)
                        if new_diffs & diffs_A:
                            continue
                        B.add(x)
                        sums_B.update(new_sums)
                        diffs_B.update(new_diffs)
                        break
                    else:  # target == 'A'
                        if len(A) >= 64:
                            continue
                        new_sums = {x + s for s in A}
                        new_sums.add(2 * x)
                        if new_sums & sums_A:
                            continue
                        new_diffs = set()
                        for s in A:
                            new_diffs.add(x - s)
                            new_diffs.add(s - x)
                        if new_diffs & diffs_B:
                            continue
                        A.add(x)
                        sums_A.update(new_sums)
                        diffs_A.update(new_diffs)
                        break
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_add(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """Try to add more elements via local search."""
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if x in A or x in B:
                    continue
                
                # Try adding to A
                if len(A) < 64:
                    new_sums = {x + s for s in A}
                    new_sums.add(2 * x)
                    if not (new_sums & sums_A):
                        new_diffs = set()
                        for s in A:
                            new_diffs.add(x - s)
                            new_diffs.add(s - x)
                        if not (new_diffs & diffs_B):
                            A.add(x)
                            sums_A.update(new_sums)
                            diffs_A.update(new_diffs)
                            improved = True
                            break
                
                # Try adding to B
                if len(B) < 64:
                    new_sums = {x + s for s in B}
                    new_sums.add(2 * x)
                    if not (new_sums & sums_B):
                        new_diffs = set()
                        for s in B:
                            new_diffs.add(x - s)
                            new_diffs.add(s - x)
                        if not (new_diffs & diffs_A):
                            B.add(x)
                            sums_B.update(new_sums)
                            diffs_B.update(new_diffs)
                            improved = True
                            break
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_swap(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """Try swapping elements between A and B, or removing/re-adding to improve."""
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        improved = True
        max_iters = 100
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try moving x from A to B
            for x in list(A):
                if x in B:
                    continue
                if len(B) >= 64:
                    continue
                
                # Remove x from A
                A_temp = A - {x}
                # Recompute sums_A_temp and diffs_A_temp
                sums_A_temp = get_sums(list(A_temp))
                diffs_A_temp = get_diffs(list(A_temp))
                
                # Try adding x to B
                new_sums = {x + s for s in B}
                new_sums.add(2 * x)
                if new_sums & sums_B:
                    continue
                new_diffs = set()
                for s in B:
                    new_diffs.add(x - s)
                    new_diffs.add(s - x)
                if new_diffs & diffs_A_temp:
                    continue
                
                # Valid move
                A = A_temp
                B.add(x)
                sums_A = sums_A_temp
                diffs_A = diffs_A_temp
                sums_B.update(new_sums)
                diffs_B.update(new_diffs)
                improved = True
                break
            
            if improved:
                continue
            
            # Try moving x from B to A
            for x in list(B):
                if x in A:
                    continue
                if len(A) >= 64:
                    continue
                
                B_temp = B - {x}
                sums_B_temp = get_sums(list(B_temp))
                diffs_B_temp = get_diffs(list(B_temp))
                
                new_sums = {x + s for s in A}
                new_sums.add(2 * x)
                if new_sums & sums_A:
                    continue
                new_diffs = set()
                for s in A:
                    new_diffs.add(x - s)
                    new_diffs.add(s - x)
                if new_diffs & diffs_B_temp:
                    continue
                
                B = B_temp
                A.add(x)
                sums_B = sums_B_temp
                diffs_B = diffs_B_temp
                sums_A.update(new_sums)
                diffs_A.update(new_diffs)
                improved = True
                break
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    def local_search_replace(A, B, sums_A, diffs_A, sums_B, diffs_B):
        """Try removing an element and replacing with a different one to unlock additions."""
        A = set(A)
        B = set(B)
        sums_A = set(sums_A)
        diffs_A = set(diffs_A)
        sums_B = set(sums_B)
        diffs_B = set(diffs_B)
        
        improved = False
        current_size = len(A) + len(B)
        
        # Try removing each element from A and see if we can add 2+ new elements
        for x in list(A):
            if len(A) <= 1:
                continue
            A_temp = A - {x}
            sums_A_temp = get_sums(list(A_temp))
            diffs_A_temp = get_diffs(list(A_temp))
            
            # Try to greedily fill up with new elements
            A_new = set(A_temp)
            B_new = set(B)
            sums_A_new = set(sums_A_temp)
            diffs_A_new = set(diffs_A_temp)
            sums_B_new = set(sums_B)
            diffs_B_new = set(diffs_B)
            
            # Try adding all possible elements
            for y in range(1, n + 1):
                if y in A_new or y in B_new:
                    continue
                # Try A_new
                if len(A_new) < 64:
                    new_sums = {y + s for s in A_new}
                    new_sums.add(2 * y)
                    if not (new_sums & sums_A_new):
                        new_diffs = set()
                        for s in A_new:
                            new_diffs.add(y - s)
                            new_diffs.add(s - y)
                        if not (new_diffs & diffs_B_new):
                            A_new.add(y)
                            sums_A_new.update(new_sums)
                            diffs_A_new.update(new_diffs)
                            continue
                # Try B_new
                if len(B_new) < 64:
                    new_sums = {y + s for s in B_new}
                    new_sums.add(2 * y)
                    if not (new_sums & sums_B_new):
                        new_diffs = set()
                        for s in B_new:
                            new_diffs.add(y - s)
                            new_diffs.add(s - y)
                        if not (new_diffs & diffs_A_new):
                            B_new.add(y)
                            sums_B_new.update(new_sums)
                            diffs_B_new.update(new_diffs)
                            continue
            
            if len(A_new) + len(B_new) > current_size:
                A, B = A_new, B_new
                sums_A, diffs_A = sums_A_new, diffs_A_new
                sums_B, diffs_B = sums_B_new, diffs_B_new
                current_size = len(A) + len(B)
                improved = True
        
        # Same for B
        for x in list(B):
            if len(B) <= 1:
                continue
            B_temp = B - {x}
            sums_B_temp = get_sums(list(B_temp))
            diffs_B_temp = get_diffs(list(B_temp))
            
            A_new = set(A)
            B_new = set(B_temp)
            sums_A_new = set(sums_A)
            diffs_A_new = set(diffs_A)
            sums_B_new = set(sums_B_temp)
            diffs_B_new = set(diffs_B_temp)
            
            for y in range(1, n + 1):
                if y in A_new or y in B_new:
                    continue
                if len(A_new) < 64:
                    new_sums = {y + s for s in A_new}
                    new_sums.add(2 * y)
                    if not (new_sums & sums_A_new):
                        new_diffs = set()
                        for s in A_new:
                            new_diffs.add(y - s)
                            new_diffs.add(s - y)
                        if not (new_diffs & diffs_B_new):
                            A_new.add(y)
                            sums_A_new.update(new_sums)
                            diffs_A_new.update(new_diffs)
                            continue
                if len(B_new) < 64:
                    new_sums = {y + s for s in B_new}
                    new_sums.add(2 * y)
                    if not (new_sums & sums_B_new):
                        new_diffs = set()
                        for s in B_new:
                            new_diffs.add(y - s)
                            new_diffs.add(s - y)
                        if not (new_diffs & diffs_A_new):
                            B_new.add(y)
                            sums_B_new.update(new_sums)
                            diffs_B_new.update(new_diffs)
                            continue
            
            if len(A_new) + len(B_new) > current_size:
                A, B = A_new, B_new
                sums_A, diffs_A = sums_A_new, diffs_A_new
                sums_B, diffs_B = sums_B_new, diffs_B_new
                current_size = len(A) + len(B)
                improved = True
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B

    # Main search loop
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Greedy construction with randomization
        A, B, sums_A, diffs_A, sums_B, diffs_B = fast_greedy(restart)
        
        # Local search: add
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_add(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Local search: swap
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_swap(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Local search: replace
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_replace(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Final add pass
        A, B, sums_A, diffs_A, sums_B, diffs_B = local_search_add(A, B, sums_A, diffs_A, sums_B, diffs_B)
        
        # Validate
        if is_valid_state(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = A
                best_B = B

    # Final check
    if not best_A or not best_B:
        return [[1], [2]]
        
    return [list(best_A), list(best_B)]

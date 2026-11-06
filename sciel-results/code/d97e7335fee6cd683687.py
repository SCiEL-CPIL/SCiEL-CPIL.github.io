import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 90s limit (or 60s eval limit, 55 is safe)
    
    n = 256
    best_A = []
    best_B = []
    best_size = 0
    
    # Helper to check Sidon and Diff constraints for adding x to S
    # S: set of elements
    # sums_S: set of pair sums
    # diffs_S: set of nonzero differences
    # D_other: set of differences of the OTHER set
    def can_add(S, sums_S, diffs_S, D_other, x):
        # Check Sidon: new sums must not be in sums_S
        # New sums: {x + s for s in S} U {2x}
        # Check Diff Disjointness: new diffs must not be in D_other
        # New diffs: {x - s for s in S} U {s - x for s in S}
        
        # Optimization: Check sums first, usually fails fast
        for s in S:
            if x + s in sums_S:
                return False
        if 2 * x in sums_S:
            return False
            
        for s in S:
            d1 = x - s
            d2 = s - x
            if d1 in D_other or d2 in D_other:
                return False
                
        return True

    def add_to_set(S, sums_S, diffs_S, x):
        # Add x to S, update sums and diffs
        S.add(x)
        for s in S:
            if s == x:
                continue
            sums_S.add(x + s)
            diffs_S.add(x - s)
            diffs_S.add(s - x)
        sums_S.add(2 * x)

    def remove_from_set(S, sums_S, diffs_S, x):
        # Remove x from S, update sums and diffs
        S.remove(x)
        for s in S:
            if s == x:
                continue
            # Remove sums and diffs involving x
            # Note: sums and diffs might be shared by other pairs, 
            # but in a Sidon set, sums are unique. 
            # However, diffs are NOT unique in a Sidon set? 
            # Wait, "Pair sums inside each set... must be unique."
            # "Nonzero difference sets must be disjoint."
            # It does NOT say differences are unique within a set.
            # So we cannot simply remove from a set if another pair produces the same diff.
            # We need to recompute or use counts.
            # Given n=256 and max size 64, recomputing diffs is O(64^2) = 4096 ops.
            # This is fast enough for local search steps.
            pass
        # Recompute is safer for diffs.
        # For sums, they are unique, so we can remove.
        # But let's just recompute both for safety and simplicity in the local search context.
        # Actually, if we recompute, we don't need to carefully remove.
        # Let's change strategy: 
        # In local search, after a move/swap, recompute sums and diffs for the affected set.
        # This is O(k^2) where k is size of set. k<=64. 64^2=4096. Very fast.
        pass

    def recompute(S):
        sums = set()
        diffs = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
            for j in range(len(S_list)):
                if i != j:
                    diffs.add(S_list[i] - S_list[j])
        return sums, diffs

    def is_valid_state(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        sums_A, diffs_A = recompute(A)
        sums_B, diffs_B = recompute(B)
        # Check Sidon
        if len(sums_A) != len(list(combinations_with_replacement(A, 2))):
            return False
        if len(sums_B) != len(list(combinations_with_replacement(B, 2))):
            return False
        # Check Disjoint Diffs
        if diffs_A & diffs_B:
            return False
        return True

    def local_search(A, B):
        A = set(A)
        B = set(B)
        
        # Initial recompute
        sums_A, diffs_A = recompute(A)
        sums_B, diffs_B = recompute(B)
        
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                break
                
            # Generate a list of random moves to try
            # 1. Add x to A
            # 2. Add x to B
            # 3. Move x A->B
            # 4. Move x B->A
            # 5. Swap x in A with y not in A/B
            # 6. Swap x in B with y not in A/B
            
            # To keep it simple and fast, we'll pick a random operation type and element.
            # Or just iterate through candidates in random order.
            
            # Let's try adds first, then moves, then swaps.
            
            # Adds
            for x in random.sample(range(1, n+1), 20): # Sample 20 random candidates
                if x in A or x in B:
                    continue
                # Try Add to A
                if len(A) < 64 and can_add(A, sums_A, diffs_A, diffs_B, x):
                    add_to_set(A, sums_A, diffs_A, x)
                    improved = True
                    break
                # Try Add to B
                if len(B) < 64 and can_add(B, sums_B, diffs_B, diffs_A, x):
                    add_to_set(B, sums_B, diffs_B, x)
                    improved = True
                    break
            
            if improved:
                # Recompute if we want to be safe? 
                # add_to_set updates sums/diffs correctly for Sidon sums (unique).
                # For diffs, it adds new ones. It doesn't remove old ones if they become invalid?
                # No, adding an element doesn't make old diffs invalid.
                # So add_to_set is correct.
                continue
                
            # Moves
            for x in list(A):
                if len(B) >= 64:
                    continue
                # Remove x from A
                A_temp = A - {x}
                # Check if x can be added to B given A_temp
                # We need diffs of A_temp.
                # Let's just try it:
                # Remove x from A properly
                # Since we don't have a remove function that handles diff multiplicity,
                # let's just simulate:
                # Can we add x to B?
                # Condition: can_add(B, sums_B, diffs_B, diffs_A_temp, x)
                # And Sidon for B: can_add_sidon(B, sums_B, x)
                
                # To get diffs_A_temp, we'd need to recompute or subtract.
                # Let's recompute A_temp diffs for the check.
                # This is O(|A|^2).
                _, diffs_A_temp = recompute(A_temp)
                
                if can_add(B, sums_B, diffs_B, diffs_A_temp, x):
                    # Execute move
                    # Remove x from A
                    A.remove(x)
                    # We need to update sums_A and diffs_A.
                    # Easiest: recompute A
                    sums_A, diffs_A = recompute(A)
                    # Add x to B
                    add_to_set(B, sums_B, diffs_B, x)
                    improved = True
                    break
                    
            if improved:
                continue
                
            # Swaps (A)
            # Pick x in A, y not in A/B
            # Remove x from A, Add y to A
            for _ in range(10): # Limit swaps tried per iteration
                if not A:
                    break
                x = random.choice(list(A))
                y_candidates = [y for y in range(1, n+1) if y not in A and y not in B]
                if not y_candidates:
                    break
                y = random.choice(y_candidates)
                
                A_temp = A - {x}
                # Check if y can be added to A_temp
                _, diffs_A_temp = recompute(A_temp)
                sums_A_temp, _ = recompute(A_temp)
                
                if can_add(A_temp, sums_A_temp, diffs_A_temp, diffs_B, y):
                    # Execute swap
                    A.remove(x)
                    sums_A, diffs_A = recompute(A)
                    add_to_set(A, sums_A, diffs_A, y)
                    improved = True
                    break
                    
            if improved:
                continue
                
            # Swaps (B)
            for _ in range(10):
                if not B:
                    break
                x = random.choice(list(B))
                y_candidates = [y for y in range(1, n+1) if y not in A and y not in B]
                if not y_candidates:
                    break
                y = random.choice(y_candidates)
                
                B_temp = B - {x}
                sums_B_temp, diffs_B_temp = recompute(B_temp)
                
                if can_add(B_temp, sums_B_temp, diffs_B_temp, diffs_A, y):
                    # Execute swap
                    B.remove(x)
                    sums_B, diffs_B = recompute(B)
                    add_to_set(B, sums_B, diffs_B, y)
                    improved = True
                    break
                    
        return list(A), list(B)

    # Greedy Randomized Adaptive Search Procedure (GRASP)
    num_restarts = 50
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        A = set()
        B = set()
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        # Randomized Greedy Construction
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
                
            # Try to add to A
            if len(A) < 64 and can_add(A, sums_A, diffs_A, diffs_B, x):
                add_to_set(A, sums_A, diffs_A, x)
                continue
            
            # Try to add to B
            if len(B) < 64 and can_add(B, sums_B, diffs_B, diffs_A, x):
                add_to_set(B, sums_B, diffs_B, x)
                continue
            
        # Local Search
        A, B = local_search(A, B)
        
        # Validate
        if is_valid_state(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = list(A)
                best_B = list(B)

    # Final check
    if not best_A or not best_B:
        return [[1], [2]]
        
    return [best_A, best_B]

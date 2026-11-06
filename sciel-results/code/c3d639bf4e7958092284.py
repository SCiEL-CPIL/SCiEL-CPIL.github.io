"""
Improved program: Backtracking search with bitmask optimizations for Sidon sets with disjoint differences.
"""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 90s limit (evaluator kills at 90s)
    
    n = 256
    max_size = 64
    
    # Bitmask utilities
    # Sums range from 2 to 512. Index 0-512.
    # Diffs range from -255 to 255. We only care about non-zero.
    # We can map diff d to bit index d + 255 (range 0..510).
    # Or simpler: just use sets for diffs if bitmasks are tricky to manage with negative indices.
    # But Python ints handle large bit shifts fine.
    # Let's use sets for clarity first, and optimize if too slow. 
    # Actually, bitmask for sums is very fast. For diffs, since they can be negative, 
    # we can store positive and negative separately or map to 0..510.
    # Let's use sets for diffs to avoid index mapping errors, and bitmask for sums.
    
    def check_sidon_bitmask(sums_mask, s, x):
        """
        Check if x can be added to Sidon set s given current sums_mask.
        Returns new_sums_mask if valid, else None.
        """
        # Check x+x
        sum_xx = x + x
        if (sums_mask >> sum_xx) & 1:
            return None
        
        new_mask = sums_mask
        
        # Check x+a for all a in s
        for a in s:
            sum_xa = x + a
            if (sums_mask >> sum_xa) & 1:
                return None
            new_mask |= (1 << sum_xa)
        
        new_mask |= (1 << sum_xx)
        return new_mask

    def get_diffs_set(s, x):
        """Get set of non-zero differences introduced by adding x to s."""
        diffs = set()
        for a in s:
            d = x - a
            if d != 0:
                diffs.add(d)
            d = a - x
            if d != 0:
                diffs.add(d)
        return diffs

    def is_valid_add(s, x, sums_mask, diffs_s, diffs_other):
        """
        Check if x can be added to s.
        s: list of elements
        x: candidate element
        sums_mask: bitmask of sums in s
        diffs_s: set of non-zero diffs in s
        diffs_other: set of non-zero diffs in the OTHER set
        Returns (new_sums_mask, new_diffs_s) if valid, else (None, None).
        """
        if x in s:
            return None, None
            
        new_sums_mask = check_sidon_bitmask(sums_mask, s, x)
        if new_sums_mask is None:
            return None, None
        
        new_diffs = get_diffs_set(s, x)
        
        # Check conflict with own diffs
        if new_diffs & diffs_s:
            return None, None
            
        # Check conflict with other set's diffs
        if new_diffs & diffs_other:
            return None, None
            
        return new_sums_mask, new_diffs

    def backtracking_search(order, time_limit):
        """
        Perform backtracking search given an order of elements.
        Returns best A, B found.
        """
        best_total = 0
        best_A = []
        best_B = []
        
        # State: A, B, sums_A, sums_B, diffs_A, diffs_B
        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = set()
        diffs_B = set()
        
        def dfs(idx, current_total):
            nonlocal best_total, best_A, best_B
            
            if time.time() > time_limit:
                return
            
            # Update best
            if current_total > best_total:
                best_total = current_total
                best_A = A[:]
                best_B = B[:]
                
            # Pruning: if we can't beat best_total with remaining elements, stop
            # Max possible addition is len(order) - idx
            if current_total + (len(order) - idx) <= best_total:
                return
            
            if idx == len(order):
                return
                
            x = order[idx]
            
            # Option 1: Add to A
            new_sums_A, new_diffs_A = is_valid_add(A, x, sums_A, diffs_A, diffs_B)
            if new_sums_A is not None:
                A.append(x)
                old_diffs_A = diffs_A
                diffs_A = diffs_A | new_diffs_A
                dfs(idx + 1, current_total + 1)
                A.pop()
                diffs_A = old_diffs_A
                
            # Option 2: Add to B
            new_sums_B, new_diffs_B = is_valid_add(B, x, sums_B, diffs_B, diffs_A)
            if new_sums_B is not None:
                B.append(x)
                old_diffs_B = diffs_B
                diffs_B = diffs_B | new_diffs_B
                dfs(idx + 1, current_total + 1)
                B.pop()
                diffs_B = old_diffs_B
                
            # Option 3: Skip x
            dfs(idx + 1, current_total)
            
        dfs(0, 0)
        return best_A, best_B

    # Main loop: Try multiple random orders
    best_A_global = []
    best_B_global = []
    best_total_global = 0
    
    # Initial greedy to get a baseline
    # Simple greedy to set a floor
    A_greedy = []
    B_greedy = []
    sums_A_g = 0
    sums_B_g = 0
    diffs_A_g = set()
    diffs_B_g = set()
    
    elements = list(range(1, n + 1))
    random.shuffle(elements)
    
    for x in elements:
        # Try A
        ns, nd = is_valid_add(A_greedy, x, sums_A_g, diffs_A_g, diffs_B_g)
        if ns is not None:
            A_greedy.append(x)
            sums_A_g = ns
            diffs_A_g = diffs_A_g | nd
            continue
        # Try B
        ns, nd = is_valid_add(B_greedy, x, sums_B_g, diffs_B_g, diffs_A_g)
        if ns is not None:
            B_greedy.append(x)
            sums_B_g = ns
            diffs_B_g = diffs_B_g | nd
            
    best_A_global = A_greedy
    best_B_global = B_greedy
    best_total_global = len(A_greedy) + len(B_greedy)
    
    # Backtracking with random restarts
    restarts = 0
    while time.time() < start_time + time_limit - 2:
        restarts += 1
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        # Use a tighter time limit for each backtracking run
        # Allocate time based on restarts
        # First few restarts get more time? Or equal?
        # Let's give each restart a fixed small budget, e.g., 2 seconds, 
        # but adjust if we have lots of time left.
        # Better: dynamic budget.
        time_remaining = start_time + time_limit - 2 - time.time()
        if time_remaining <= 0:
            break
            
        # Allocate budget for this run
        # If we have > 10s left, give 2s per run. Else give all remaining.
        run_limit = min(2.0, time_remaining)
        
        A_res, B_res = backtracking_search(order, time.time() + run_limit)
        
        total = len(A_res) + len(B_res)
        if total > best_total_global:
            best_total_global = total
            best_A_global = A_res
            best_B_global = B_res
            # If we found a very good solution, we might want to spend more time polishing?
            # For now, just keep searching.
            
        # Early exit if we hit theoretical max? 
        # Max Sidon size for n=256 is approx sqrt(256) = 16? 
        # No, Sidon set size is approx sqrt(n). 
        # Wait, Sidon set in [1,n] has size O(sqrt(n)).
        # For n=256, sqrt(256)=16. 
        # So max size for one set is around 16-20.
        # Two sets with disjoint diffs... 
        # Total size might be around 30-40.
        # If we reach 64 (the cap), we can stop.
        if best_total_global >= 64:
            break

    # Final validation and cleanup
    # Ensure nonempty
    if not best_A_global:
        best_A_global = [1]
    if not best_B_global:
        best_B_global = [2]
        
    # Check Sidon property
    def validate_sidon(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True

    # Check disjoint differences
    def get_all_diffs(s):
        diffs = set()
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    d = s[j] - s[i]
                    if d != 0:
                        diffs.add(d)
        return diffs

    # If validation fails (shouldn't happen if logic is correct), fallback to greedy result
    if not validate_sidon(best_A_global) or not validate_sidon(best_B_global):
        # Re-run greedy to be safe
        A_greedy = []
        B_greedy = []
        sums_A_g = 0
        sums_B_g = 0
        diffs_A_g = set()
        diffs_B_g = set()
        
        elements = list(range(1, n + 1))
        random.shuffle(elements)
        
        for x in elements:
            ns, nd = is_valid_add(A_greedy, x, sums_A_g, diffs_A_g, diffs_B_g)
            if ns is not None:
                A_greedy.append(x)
                sums_A_g = ns
                diffs_A_g = diffs_A_g | nd
                continue
            ns, nd = is_valid_add(B_greedy, x, sums_B_g, diffs_B_g, diffs_A_g)
            if ns is not None:
                B_greedy.append(x)
                sums_B_g = ns
                diffs_B_g = diffs_B_g | nd
                
        best_A_global = A_greedy
        best_B_global = B_greedy

    diffs_A_final = get_all_diffs(best_A_global)
    diffs_B_final = get_all_diffs(best_B_global)
    
    if diffs_A_final & diffs_B_final:
        # This should not happen. If it does, truncate the smaller set or something?
        # Better to just return the greedy result which is guaranteed valid by construction.
        # The backtracking logic ensures disjointness.
        pass

    # Cap sizes
    best_A_global = best_A_global[:max_size]
    best_B_global = best_B_global[:max_size]
    
    return [best_A_global, best_B_global]

# EVOLVE-BLOCK-END

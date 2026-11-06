"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave some buffer for 90s limit

    # Theoretical bound: for |S|=n, number of triples is n(n+1)(n+2)/6.
    # Sums range from 3 to 12288, giving 12286 possible distinct sums.
    # n=40: 11480 <= 12286. OK.
    # n=41: 12411 > 12286. Not OK.
    # So the theoretical maximum is 40.
    
    def verify(s):
        """Verify that the set has unique triple sums."""
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 4096 for x in s):
            return False
        sums = [sum(t) for t in combinations_with_replacement(s, 3)]
        return len(sums) == len(set(sums))

    def run_greedy_fast(order, time_budget, start_time_ref, initial_selected=None):
        """Fast greedy using numpy for vectorized collision checks."""
        if initial_selected is None:
            initial_selected = []
            
        selected = list(initial_selected)
        if len(selected) == 0:
            sums_mask = np.zeros(3 * 4096 + 1, dtype=bool)
        else:
            # Initialize mask from initial_selected
            sums_mask = np.zeros(3 * 4096 + 1, dtype=bool)
            sel_arr = np.array(selected, dtype=np.int32)
            # Compute all triple sums for initial set
            # This is O(N^3) but N is small initially or we can just rely on the fact that 
            # we will only add to it. But for correctness of the mask, we need it.
            # Actually, if initial_selected is provided, we should have precomputed sums.
            # For simplicity in this edit, let's assume initial_selected is empty or 
            # we recompute. Recomputing for N=40 is 11480 sums, which is fast.
            for t in combinations_with_replacement(sel_arr, 3):
                s = int(t[0] + t[1] + t[2])
                sums_mask[s] = True
        
        # Pre-compute pairs of selected elements for faster lookup?
        # No, we can just use the mask.
        
        selected_arr = np.array(selected, dtype=np.int32)
        
        for x in order:
            if time.time() - start_time_ref > time_budget:
                break
            if len(selected) >= 40:
                break
            
            x_int = int(x)
            
            # Generate all new sums involving x
            # Types: (x,x,x), (x,x,y), (x,y,y), (x,y,z)
            
            # (x,x,x)
            s_0 = 3 * x_int
            if sums_mask[s_0]:
                continue
            
            # (x,x,y) for y in selected
            if len(selected_arr) > 0:
                sums_xx_y = 2 * x_int + selected_arr
                # Check if any are in mask
                if np.any(sums_mask[sums_xx_y]):
                    continue
            
            # (x,y,y) for y in selected
            if len(selected_arr) > 0:
                sums_x_yy = x_int + 2 * selected_arr
                if np.any(sums_mask[sums_x_yy]):
                    continue
            
            # (x,y,z) for y < z in selected
            if len(selected_arr) > 1:
                n = len(selected_arr)
                # We need to check all pairs y, z from selected_arr
                # Vectorizing this is tricky without loops, but N is small.
                # Let's loop over pairs but check against mask quickly.
                valid = True
                for i in range(n):
                    y = selected_arr[i]
                    # z must be > y to avoid duplicates? No, just distinct pairs in set.
                    # In combinations_with_replacement, order doesn't matter.
                    # So we check all pairs (y, z) with y != z? 
                    # Actually, (x, y, z) with y, z in S.
                    # We need to ensure (x, y, z) sum is not in mask.
                    # We also need to ensure we don't double count or miss.
                    # The mask contains all sums of triples from *existing* S.
                    # So we just check if x+y+z is in mask.
                    # We also need to add these sums to mask later.
                    
                    # Optimization: Check all sums x + y + z for y, z in S.
                    # This is x + S + S.
                    # We can vectorize: sums_x_yz = x_int + selected_arr[:, None] + selected_arr[None, :]
                    # But we only need y < z? No, the sum is the same.
                    # We just need to check if ANY of these sums are in the mask.
                    # And we need to add them to the mask.
                    
                    # Let's just compute the full set of new sums for x.
                    pass # We'll do a unified check below to be safe and simple.
            
            # Unified check and update
            new_sums_to_check = []
            new_sums_to_add = set()
            
            # 1. (x, x, x)
            s = 3 * x_int
            new_sums_to_check.append(s)
            new_sums_to_add.add(s)
            
            if len(selected_arr) > 0:
                # 2. (x, x, y)
                sums_xx_y = 2 * x_int + selected_arr
                new_sums_to_check.extend(sums_xx_y.tolist())
                new_sums_to_add.update(sums_xx_y.tolist())
                
                # 3. (x, y, y)
                sums_x_yy = x_int + 2 * selected_arr
                new_sums_to_check.extend(sums_x_yy.tolist())
                new_sums_to_add.update(sums_x_yy.tolist())
                
                # 4. (x, y, z)
                # We need to check all pairs (y, z) from selected_arr.
                # Since selected_arr is small, we can loop.
                n = len(selected_arr)
                for i in range(n):
                    for j in range(i + 1, n):
                        s = x_int + selected_arr[i] + selected_arr[j]
                        new_sums_to_check.append(s)
                        new_sums_to_add.add(s)
            
            # Check collisions
            check_arr = np.array(new_sums_to_check, dtype=np.int32)
            if np.any(sums_mask[check_arr]):
                continue
            
            # Accept x
            # Update mask
            sums_mask[list(new_sums_to_add)] = True
            selected.append(x_int)
            selected_arr = np.append(selected_arr, x_int)
        
        return selected

    # Strategy 1: Fast greedy from 1 upward
    best_set = run_greedy_fast(range(1, 4097), time_limit - (time.time() - start_time), start_time)
    
    # Strategy 2: Fast greedy from larger numbers downward
    if time.time() - start_time < time_limit - 10:
        s2 = run_greedy_fast(range(4096, 0, -1), time_limit - (time.time() - start_time), start_time)
        if len(s2) > len(best_set):
            best_set = s2
    
    # Strategy 3: Random orderings with different seeds
    random.seed(42)
    for trial in range(50):
        if time.time() - start_time > time_limit - 5:
            break
        order = list(range(1, 4097))
        random.shuffle(order)
        s = run_greedy_fast(order, time_limit - (time.time() - start_time), start_time)
        if len(s) > len(best_set):
            best_set = s
            
    # Strategy 4: Extend best set
    if time.time() - start_time < time_limit - 10 and len(best_set) < 40:
        base = best_set[:]
        base_set = set(base)
        remaining = [x for x in range(1, 4097) if x not in base_set]
        # Sort remaining to try small ones first? Or random?
        # Let's try small ones first as they are more likely to fit?
        # Or just use the fast greedy on remaining, initializing with base.
        extended = run_greedy_fast(remaining, time_limit - (time.time() - start_time), start_time, initial_selected=base)
        if len(extended) > len(base):
            best_set = extended

    # Verify the result
    if not verify(best_set):
        # Fallback to a simple valid set
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    return best_set
# EVOLVE-BLOCK-END

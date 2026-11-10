import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin
    
    n = 128
    targets = list(range(128, 257))
    target_mask = np.zeros(257, dtype=bool)
    target_mask[128:257] = True
    
    # A is a set of integers
    # counts is an array of length 257
    # We start with A = {0, 1, ..., 128}
    A = set(range(n + 1))
    
    def compute_counts(A_set):
        counts = np.zeros(257, dtype=int)
        A_list = sorted(A_set)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if s <= 256:
                    counts[s] += 1
        return counts

    def get_peak_and_sq(counts):
        # Only care about 128..256
        c = counts[128:257]
        peak = np.max(c)
        sq = np.sum(c**2)
        return peak, sq

    def is_covered(counts):
        return np.all(counts[128:257] > 0)

    # Initial counts
    counts = compute_counts(A)
    curr_peak, curr_sq = get_peak_and_sq(counts)
    best_peak = curr_peak
    best_sq = curr_sq
    best_A = set(A)
    
    # Helper to update counts when adding/removing x
    # We will simulate moves by creating temporary counts or updating in place and reverting.
    # Given the small size, copying the array (size 257) is cheap.
    
    def try_move(A_set, x, y, counts_arr):
        """
        Simulate removing x and adding y.
        Returns (new_peak, new_sq, is_valid)
        """
        # If x == y, no change
        if x == y:
            return curr_peak, curr_sq, True
            
        new_counts = counts_arr.copy()
        
        # Remove x
        if x in A_set:
            A_set.remove(x)
            # Decrement pairs (x, z) for z in A_set
            # Note: A_set is already modified (x removed)
            for z in A_set:
                s = x + z
                if s <= 256:
                    new_counts[s] -= 1
            s = 2 * x
            if s <= 256:
                new_counts[s] -= 1
        
        # Add y
        if y not in A_set:
            A_set.add(y)
            # Increment pairs (y, z) for z in A_set
            # Note: A_set includes y now? 
            # We added y to A_set.
            # Pairs are (y, z) for z in A_set, z != y? 
            # Unordered pairs (a,b) with a<=b.
            # When adding y, new pairs are (y, z) where z in A_set.
            # If z == y, it's (y,y). If z != y, it's (y,z) or (z,y).
            # Our loop `for z in A_set` includes y.
            # If z == y, s = 2y.
            # If z != y, s = y+z.
            # This correctly counts all new pairs involving y.
            for z in A_set:
                s = y + z
                if s <= 256:
                    new_counts[s] += 1
            # Wait, if y was not in A_set, we added it.
            # The loop iterates over the NEW A_set.
            # This counts (y, y) once (when z=y) and (y, z) for z!=y once.
            # This is correct for unordered pairs.
            
            # Restore A_set to original state for the caller? 
            # No, this function is a simulation. We should not modify the actual A_set passed in 
            # unless we are committing the move.
            # Let's refactor: pass a copy of A_set or just calculate deltas.
            # Given the complexity, let's just use a local set copy in the simulation.
            
        # Check coverage
        if not np.all(new_counts[128:257] > 0):
            return float('inf'), float('inf'), False
            
        peak, sq = get_peak_and_sq(new_counts)
        return peak, sq, True

    # The above try_move modifies A_set. We need to be careful.
    # Let's rewrite the loop to be cleaner.
    
    A_list = list(A)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try Removals (x in A, y = None)
        # We iterate through elements of A
        A_snapshot = list(A)
        random.shuffle(A_snapshot)
        
        best_rem_peak = curr_peak
        best_rem_sq = curr_sq
        best_rem_x = None
        best_rem_counts = None
        
        for x in A_snapshot:
            # Simulate removing x
            new_counts = counts.copy()
            
            # Decrement pairs involving x
            for z in A:
                if z == x:
                    continue
                s = x + z
                if s <= 256:
                    new_counts[s] -= 1
            
            s = 2 * x
            if s <= 256:
                new_counts[s] -= 1
                
            if not np.all(new_counts[128:257] > 0):
                continue
                
            peak, sq = get_peak_and_sq(new_counts)
            
            # Accept if peak is lower, or peak same and sq lower
            if peak < best_rem_peak or (peak == best_rem_peak and sq < best_rem_sq):
                best_rem_peak = peak
                best_rem_sq = sq
                best_rem_x = x
                best_rem_counts = new_counts
        
        if best_rem_x is not None:
            # Apply removal
            A.remove(best_rem_x)
            counts = best_rem_counts
            curr_peak = best_rem_peak
            curr_sq = best_rem_sq
            improved = True
            if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                best_peak = curr_peak
                best_sq = curr_sq
                best_A = set(A)
            continue # Restart loop with new state
        
        # 2. Try Swaps (x in A, y not in A)
        # To save time, we might not try ALL swaps if removals are frequent.
        # But if stuck, we must try swaps.
        
        # Optimization: Only try swaps if we haven't improved in a while or always?
        # Let's try a limited number of random swaps if full search is too slow, 
        # but 129*129 is small enough.
        
        A_snapshot = list(A)
        candidates_out = [i for i in range(n + 1) if i not in A]
        
        if not candidates_out:
            # A is full, no swaps possible
            break
            
        best_swap_peak = curr_peak
        best_swap_sq = curr_sq
        best_swap_x = None
        best_swap_y = None
        best_swap_counts = None
        
        # Shuffle to avoid bias
        random.shuffle(A_snapshot)
        random.shuffle(candidates_out)
        
        # We can limit the number of swaps checked per iteration if it's too slow
        # But let's try all first.
        
        for x in A_snapshot:
            for y in candidates_out:
                new_counts = counts.copy()
                
                # Remove x
                for z in A:
                    if z == x:
                        continue
                    s = x + z
                    if s <= 256:
                        new_counts[s] -= 1
                s = 2 * x
                if s <= 256:
                    new_counts[s] -= 1
                    
                # Add y
                for z in A:
                    # z is from original A. 
                    # The new pairs are (y, z) for z in A, plus (y, y) if y not in A (which is true).
                    # Wait, if we remove x and add y, the set is (A - {x}) U {y}.
                    # Pairs involving y are (y, z) where z in (A - {x}) U {y}.
                    # So z can be any element in A except x, AND y itself.
                    
                    if z == x:
                        continue # x is removed
                    s = y + z
                    if s <= 256:
                        new_counts[s] += 1
                # Add (y, y)
                s = 2 * y
                if s <= 256:
                    new_counts[s] += 1
                    
                if not np.all(new_counts[128:257] > 0):
                    continue
                
                peak, sq = get_peak_and_sq(new_counts)
                
                if peak < best_swap_peak or (peak == best_swap_peak and sq < best_swap_sq):
                    best_swap_peak = peak
                    best_swap_sq = sq
                    best_swap_x = x
                    best_swap_y = y
                    best_swap_counts = new_counts
                    
        if best_swap_x is not None:
            # Apply swap
            A.remove(best_swap_x)
            A.add(best_swap_y)
            counts = best_swap_counts
            curr_peak = best_swap_peak
            curr_sq = best_swap_sq
            improved = True
            if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                best_peak = curr_peak
                best_sq = curr_sq
                best_A = set(A)
            continue # Restart loop

        # 3. If no improvement, try a random perturbation to escape local minimum
        # Swap 2 random elements
        if len(A) > 2:
            x1, x2 = random.sample(list(A), 2)
            cands = [i for i in range(n + 1) if i not in A]
            if len(cands) >= 2:
                y1, y2 = random.sample(cands, 2)
                
                new_counts = counts.copy()
                # Remove x1, x2
                for x in [x1, x2]:
                    for z in A:
                        if z == x: continue
                        s = x + z
                        if s <= 256: new_counts[s] -= 1
                    s = 2 * x
                    if s <= 256: new_counts[s] -= 1
                # Note: The pair (x1, x2) was counted in both loops?
                # Loop 1 (x=x1): z=x2 -> s=x1+x2. Decrement.
                # Loop 2 (x=x2): z=x1 -> s=x2+x1. Decrement.
                # So (x1,x2) is decremented twice? 
                # In the original counts, (x1,x2) was counted once.
                # So we must be careful.
                
                # Let's just recompute counts for perturbation to be safe, it's rare.
                A_trial = set(A)
                A_trial.remove(x1)
                A_trial.remove(x2)
                A_trial.add(y1)
                A_trial.add(y2)
                
                new_counts = compute_counts(A_trial)
                
                if np.all(new_counts[128:257] > 0):
                    peak, sq = get_peak_and_sq(new_counts)
                    # Accept perturbation with some probability or if it's not terrible
                    # For now, accept if it's valid.
                    A = A_trial
                    counts = new_counts
                    curr_peak = peak
                    curr_sq = sq
                    # Don't update best unless it's better
                    if curr_peak < best_peak or (curr_peak == best_peak and curr_sq < best_sq):
                        best_peak = curr_peak
                        best_sq = curr_sq
                        best_A = set(A)

    return sorted(list(best_A))

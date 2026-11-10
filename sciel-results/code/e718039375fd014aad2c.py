import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    n = 128
    targets = list(range(128, 257))
    
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
        c = counts[128:257]
        peak = np.max(c)
        sq = np.sum(c**2)
        return peak, sq

    def is_covered(counts):
        return np.all(counts[128:257] > 0)

    # 1. Greedy Initialization to ensure coverage
    # Start with a small set and grow it to cover [128, 256]
    A = set()
    uncovered = set(targets)
    
    # We need to cover [128, 256].
    # A simple greedy: pick x that maximizes new coverage.
    while uncovered:
        best_x = -1
        best_new_cov = 0
        
        for x in range(n + 1):
            if x in A:
                continue
            # How many uncovered targets does x help cover?
            # If x is added, new pairs are (x, y) for y in A, and (x, x).
            # Sums are x+y and 2x.
            new_sums = set()
            for a in A:
                new_sums.add(x + a)
            new_sums.add(2*x)
            
            cov = len(new_sums & uncovered)
            if cov > best_new_cov:
                best_new_cov = cov
                best_x = x
        
        if best_x == -1 or best_new_cov == 0:
            # If stuck, pick a random candidate or break
            # This shouldn't happen often if A is non-empty and targets are contiguous
            if A:
                # Try to find any x that covers at least 1 uncovered
                for x in range(n + 1):
                    if x in A: continue
                    new_sums = set()
                    for a in A:
                        new_sums.add(x + a)
                    new_sums.add(2*x)
                    if new_sums & uncovered:
                        best_x = x
                        break
                if best_x == -1:
                    break
            else:
                best_x = 128 # 256 is a target, 128+128=256

        A.add(best_x)
        
        # Update uncovered
        new_sums = set()
        for a in A:
            new_sums.add(best_x + a)
        new_sums.add(2*best_x)
        uncovered -= new_sums

    if not A:
        return [128]

    # 2. Local Search
    counts = compute_counts(A)
    curr_peak, curr_sq = get_peak_and_sq(counts)
    best_peak = curr_peak
    best_sq = curr_sq
    best_A = set(A)
    
    # If initial greedy didn't cover everything, try to fix it or fallback to full range
    if not is_covered(counts):
        # Fallback: start with full range and remove
        A = set(range(n + 1))
        counts = compute_counts(A)
        curr_peak, curr_sq = get_peak_and_sq(counts)
        best_peak = curr_peak
        best_sq = curr_sq
        best_A = set(A)

    # Precompute membership for faster checks
    A_list = list(A)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try Removals
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
            # Pairs (x, z) for z in A, z != x, and (x, x)
            for z in A:
                if z == x:
                    continue
                s = x + z
                if s <= 256:
                    new_counts[s] -= 1
            
            s = 2 * x
            if s <= 256:
                new_counts[s] -= 1
                
            if not is_covered(new_counts):
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
        
        # 2. Try Swaps
        # To save time, we might not try ALL swaps if removals are frequent.
        # But if stuck, we must try swaps.
        
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
        # But let's try all first. 129*129 is small enough.
        
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
                # New pairs are (y, z) for z in (A - {x}) U {y}
                # i.e., z in A, z != x, AND z = y
                for z in A:
                    if z == x:
                        continue # x is removed
                    s = y + z
                    if s <= 256:
                        new_counts[s] += 1
                # Add (y, y)
                s = 2 * y
                if s <= 256:
                    new_counts[s] += 1
                    
                if not is_covered(new_counts):
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
                
                A_trial = set(A)
                A_trial.remove(x1)
                A_trial.remove(x2)
                A_trial.add(y1)
                A_trial.add(y2)
                
                new_counts = compute_counts(A_trial)
                
                if is_covered(new_counts):
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

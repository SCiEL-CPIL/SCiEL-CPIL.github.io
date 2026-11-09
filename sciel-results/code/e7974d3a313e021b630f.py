import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128
    MAX_SUM = 2 * MAX_VAL

    def calc_reps(A):
        """Compute representation counts for sums N_LOW..N_HIGH using unordered pairs.
        Uses convolution for speed."""
        if not A:
            return np.zeros(RANGE_SIZE, dtype=np.int32)
        
        ind = np.bincount(list(A), minlength=MAX_VAL + 1).astype(np.int32)
        conv = np.convolve(ind, ind, mode='full')
        conv_slice = conv[N_LOW:N_HIGH+1]
        
        s_vals = np.arange(N_LOW, N_HIGH + 1)
        half_s = s_vals // 2
        is_even = (s_vals % 2 == 0)
        diag = np.where(is_even, ind[half_s], 0)
        
        reps = (conv_slice + diag) // 2
        return reps

    def is_covered(reps):
        return np.all(reps > 0)

    def get_peak(reps):
        return int(np.max(reps))

    def update_reps_add(reps, A, x):
        """Update reps array in-place when adding x to A.
        Returns a copy of the updated reps to keep original intact for backtracking if needed, 
        but here we will manage state carefully. Actually, for speed, let's return new array or modify in place 
        and store previous state? No, better to compute delta and apply.
        """
        # Adding x to A:
        # Pairs (x, a) for a in A.
        # If x is already in A, we are adding a new element? No, A is a set. 
        # This function assumes x is NOT in A.
        # New pairs: (x, a) for all a in A.
        # Since pairs are unordered, if x > a, it's (a, x). If x < a, it's (x, a).
        # The sum is x + a.
        # Each such sum gets +1 representation.
        # Wait, if x is added, does it form pairs with itself? No, (x,x) is only formed if x was already there?
        # No, if x is NEW, (x,x) is a new pair.
        # So sums to increment:
        # 1. x + a for all a in A.
        # 2. x + x (just one sum).
        
        if not A:
            new_reps = reps.copy()
            if 2*x >= N_LOW and 2*x <= N_HIGH:
                idx = 2*x - N_LOW
                new_reps[idx] += 1
            return new_reps

        # Sums from pairs (x, a) where a in A
        # Note: if a == x, it's not in A yet.
        sums_with_a = np.array([x + a for a in A])
        # Filter sums in range
        mask = (sums_with_a >= N_LOW) & (sums_with_a <= N_HIGH)
        valid_sums = sums_with_a[mask]
        
        new_reps = reps.copy()
        if len(valid_sums) > 0:
            indices = valid_sums - N_LOW
            # np.add.at handles duplicates
            np.add.at(new_reps, indices, 1)
            
        # Pair (x, x)
        sum_xx = 2 * x
        if N_LOW <= sum_xx <= N_HIGH:
            idx = sum_xx - N_LOW
            new_reps[idx] += 1
            
        return new_reps

    def update_reps_remove(reps, A, x):
        """Update reps array when removing x from A.
        Assumes x is in A.
        """
        # Pairs removed: (x, a) for all a in A (including x).
        # If a == x, pair (x,x) is removed.
        # If a != x, pair (x,a) is removed.
        
        sums_to_dec = []
        for a in A:
            s = x + a
            if N_LOW <= s <= N_HIGH:
                sums_to_dec.append(s)
                
        new_reps = reps.copy()
        if sums_to_dec:
            indices = np.array(sums_to_dec) - N_LOW
            np.subtract.at(new_reps, indices, 1)
            
        return new_reps

    # --- Phase 1: Greedy Construction with Fast Updates ---
    # Start with a random small set or empty?
    # Let's start with a random subset to break symmetry
    A_current = set()
    reps = np.zeros(RANGE_SIZE, dtype=np.int32)
    
    # Add a few random elements to start
    for _ in range(10):
        x = random.randint(0, MAX_VAL)
        if x not in A_current:
            A_current.add(x)
            reps = update_reps_add(reps, A_current - {x}, x) # Careful with state
    
    # Actually, simpler to just use calc_reps for init and then use deltas.
    A_current = set(random.sample(range(0, MAX_VAL+1), 20))
    reps = calc_reps(A_current)
    
    # Greedy fill
    while not is_covered(reps):
        best_x = None
        best_reps = None
        best_peak = float('inf')
        best_covered_count = -1
        
        # Sample candidates to save time
        candidates = [x for x in range(MAX_VAL + 1) if x not in A_current]
        random.shuffle(candidates)
        # Check only a subset if too many? No, 129 is small.
        
        for x in candidates:
            # Estimate gain
            # We need to see which uncovered sums are covered by x.
            # Sums involving x: x+a for a in A, and 2x.
            # Let's just compute new reps for a few promising ones?
            # To be fast, we can check coverage gain analytically.
            
            # Sums covered by adding x:
            # 1. 2x
            # 2. x+a for a in A
            
            covered_by_x = set()
            if N_LOW <= 2*x <= N_HIGH:
                covered_by_x.add(2*x)
            for a in A_current:
                s = x + a
                if N_LOW <= s <= N_HIGH:
                    covered_by_x.add(s)
            
            # Count how many of these are currently uncovered
            new_covered = 0
            for s in covered_by_x:
                if reps[s - N_LOW] == 0:
                    new_covered += 1
            
            if new_covered == 0:
                continue
                
            if new_covered > best_covered_count:
                # Compute actual peak to break ties
                # This is expensive, so only do it if we beat the current best count
                # Or just pick the first one with max new_covered?
                # Let's compute peak for the best candidate found so far in this loop?
                # No, let's just pick the one with max new_covered. If tie, random.
                best_x = x
                best_covered_count = new_covered
            elif new_covered == best_covered_count and best_x is not None:
                # Tie-breaker: prefer lower peak?
                # We'd need to compute peak for both. Expensive.
                # Let's just stick with the first one or randomize.
                if random.random() < 0.5:
                    best_x = x
                    best_covered_count = new_covered

        if best_x is None:
            # No single element helps? Should not happen if A is small enough.
            # If it does, fallback to random add
            best_x = random.choice(candidates) if candidates else None
            if best_x is None:
                break

        # Apply the move
        old_A = A_current.copy()
        A_current.add(best_x)
        new_reps = update_reps_add(reps, old_A, best_x)
        
        if not is_covered(new_reps):
            # Check if we actually improved coverage?
            # If not, revert?
            # The greedy choice was based on "new_covered" count. 
            # It's possible that adding x covers new sums but we check if it's valid.
            # Since we only care about coverage > 0, and we picked x that covers some 0s, 
            # coverage should strictly increase unless there's a bug.
            pass
            
        reps = new_reps

    if not is_covered(reps):
         # Fallback
         A_current = set(range(64, 129))
         reps = calc_reps(A_current)

    best_A = list(A_current)
    best_peak = get_peak(reps)
    
    # --- Phase 2: Removal Pass ---
    current_A = set(best_A)
    current_reps = calc_reps(current_A)
    current_peak = get_peak(current_reps)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in list(current_A):
            test_reps = update_reps_remove(current_reps, current_A, x)
            if is_covered(test_reps):
                test_peak = get_peak(test_reps)
                if test_peak <= current_peak:
                    current_A.remove(x)
                    current_reps = test_reps
                    if test_peak < current_peak:
                        current_peak = test_peak
                    improved = True
                    break # Restart loop to check other elements

    if is_covered(current_reps):
        peak = get_peak(current_reps)
        if peak < best_peak:
            best_peak = peak
            best_A = list(current_A)
            current_peak = peak
            current_reps = current_reps
        else:
            # If no improvement, keep best_A
            current_A = set(best_A)
            current_reps = calc_reps(current_A)
            current_peak = best_peak

    # --- Phase 3: Local Search (Peak Breaking) ---
    # Target: Reduce the maximum multiplicity.
    # Strategy: Find sums with max multiplicity. Try to swap an element involved in these sums
    # with a random element that doesn't hurt coverage.
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Use the current best state
    current_A = set(best_A)
    current_reps = calc_reps(current_A)
    current_peak = get_peak(current_reps)
    
    while time.time() < deadline - 5 and T > min_T:
        if current_peak <= 1:
            break
            
        # Find peak sums
        peak_indices = np.where(current_reps == current_peak)[0]
        if len(peak_indices) == 0:
            break
            
        # Pick a random peak sum
        target_idx = peak_indices[random.randint(0, len(peak_indices)-1)]
        target_sum = N_LOW + target_idx
        
        # Find pairs in A that sum to target_sum
        # a + b = target_sum, a <= b, a,b in A
        pairs = []
        A_list = list(current_A)
        for i, a in enumerate(A_list):
            b = target_sum - a
            if b in current_A and a <= b:
                pairs.append((a, b))
                
        if not pairs:
            T *= cooling_rate
            continue
            
        # Pick a random pair
        a, b = pairs[random.randint(0, len(pairs)-1)]
        
        # We want to remove one of them, say 'a', and replace with 'y'.
        # 'y' should not create new peaks and should maintain coverage.
        # Ideally, 'y' should not contribute to target_sum or any other high-multiplicity sum.
        
        best_new_A = None
        best_new_reps = None
        best_new_peak = current_peak
        
        # Try a few random replacements
        for _ in range(10):
            y = random.randint(0, MAX_VAL)
            if y == a:
                continue
                
            new_A = set(current_A)
            new_A.remove(a)
            new_A.add(y)
            
            if len(new_A) == 0:
                continue
            
            # Calculate new reps
            # Remove a, add y
            # This is two steps. 
            # 1. Remove a
            reps_no_a = update_reps_remove(current_reps, current_A, a)
            # 2. Add y
            new_reps = update_reps_add(reps_no_a, new_A - {y}, y)
            
            if is_covered(new_reps):
                new_peak = get_peak(new_reps)
                if new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_new_A = new_A
                    best_new_reps = new_reps
                elif new_peak == best_new_peak:
                    # Accept with probability? Or just keep first?
                    # Let's accept if it reduces peak or maintains it but maybe better distribution?
                    # For now, stick to strict peak reduction or equal.
                    if random.random() < 0.1: # Small chance to accept equal to escape local minima?
                         best_new_peak = new_peak
                         best_new_A = new_A
                         best_new_reps = new_reps

        if best_new_A is not None and best_new_peak < current_peak:
            current_A = best_new_A
            current_reps = best_new_reps
            current_peak = best_new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = list(current_A)
                
        T *= cooling_rate
        
    # Final check
    if best_A is None:
        return list(range(64, 129))
        
    return best_A

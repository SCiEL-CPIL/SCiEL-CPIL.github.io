import time
import random
import numpy as np
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill
    
    random.seed(42)
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = target_high - target_low + 1  # 129 targets
    
    def compute_reps_full(A_list):
        """Compute representation counts for all targets [128,256].
        A_list is a sorted list of set elements.
        Returns reps array of length 129."""
        reps = np.zeros(num_targets, dtype=np.int32)
        m = len(A_list)
        for i in range(m):
            ai = A_list[i]
            for j in range(i, m):
                s = ai + A_list[j]
                if target_low <= s <= target_high:
                    reps[s - target_low] += 1
        return reps
    
    def get_peak(reps):
        """Get peak multiplicity, return inf if any target uncovered."""
        if np.any(reps == 0):
            return float('inf')
        return int(np.max(reps))
    
    def compute_score(reps):
        """Compute a composite score: peak * 1000 + variance. Lower is better."""
        if np.any(reps == 0):
            return float('inf')
        peak = int(np.max(reps))
        mean = np.mean(reps)
        var = np.sum((reps - mean) ** 2)
        return peak * 1000.0 + var
    
    def incremental_swap_update(A_list, reps, old_val, new_val):
        """Compute new reps after swapping old_val with new_val in A_list.
        A_list is sorted. Returns (new_reps, new_peak, new_score)."""
        # Find index of old_val
        idx = A_list.index(old_val)
        
        # Build new list
        A_new = A_list[:]
        A_new[idx] = new_val
        A_new.sort()
        
        # Incremental update:
        # Remove contributions of old_val with all elements
        # Add contributions of new_val with all elements
        new_reps = reps.copy()
        
        # Remove old_val's contributions
        for a in A_list:
            s = old_val + a
            if target_low <= s <= target_high:
                new_reps[s - target_low] -= 1
        
        # Add new_val's contributions
        for a in A_new:
            s = new_val + a
            if target_low <= s <= target_high:
                new_reps[s - target_low] += 1
        
        # Check for negative values (shouldn't happen if logic is correct)
        if np.any(new_reps < 0):
            # Fallback: full recompute
            new_reps = compute_reps_full(A_new)
        
        new_peak = get_peak(new_reps)
        new_score = compute_score(new_reps)
        return A_new, new_reps, new_peak, new_score
    
    def add_element_update(A_list, reps, new_val):
        """Compute new reps after adding new_val to A_list.
        A_list is sorted. Returns (new_reps, new_peak, new_score)."""
        A_new = A_list + [new_val]
        A_new.sort()
        
        new_reps = reps.copy()
        
        # Add new_val's contributions with all elements (including itself)
        for a in A_new:
            s = new_val + a
            if target_low <= s <= target_high:
                new_reps[s - target_low] += 1
        
        if np.any(new_reps < 0):
            new_reps = compute_reps_full(A_new)
        
        new_peak = get_peak(new_reps)
        new_score = compute_score(new_reps)
        return A_new, new_reps, new_peak, new_score
    
    def remove_element_update(A_list, reps, old_val):
        """Compute new reps after removing old_val from A_list.
        A_list is sorted. Returns (new_reps, new_peak, new_score)."""
        idx = A_list.index(old_val)
        A_new = A_list[:idx] + A_list[idx+1:]
        
        new_reps = reps.copy()
        
        # Remove old_val's contributions
        for a in A_list:
            s = old_val + a
            if target_low <= s <= target_high:
                new_reps[s - target_low] -= 1
        
        if np.any(new_reps < 0):
            new_reps = compute_reps_full(A_new)
        
        new_peak = get_peak(new_reps)
        new_score = compute_score(new_reps)
        return A_new, new_reps, new_peak, new_score
    
    best_A = None
    best_peak = float('inf')
    best_score = float('inf')
    
    # Phase 1: Multi-start with different sizes and simulated annealing
    for size in [40, 45, 50, 55]:
        if time.time() > deadline:
            break
        
        max_restarts = 10 if size <= 45 else 6
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize: random sample
            A = sorted(random.sample(range(0, n + 1), size))
            reps = compute_reps_full(A)
            peak = get_peak(reps)
            score = compute_score(reps)
            
            if peak < best_peak or (peak == best_peak and score < best_score):
                best_peak = peak
                best_score = score
                best_A = list(A)
            
            # Simulated annealing with structural moves
            A_set = set(A)
            T = 5.0
            cooling = 0.9995
            
            for iteration in range(8000):
                if time.time() > deadline:
                    break
                
                # Choose move type: add, remove, or swap
                move_type = random.random()
                
                if move_type < 0.25 and len(A) < 65:
                    # Add an element
                    candidates = [x for x in range(n + 1) if x not in A_set]
                    if not candidates:
                        continue
                    new_val = random.choice(candidates)
                    A_new, new_reps, new_peak, new_score = add_element_update(A, reps, new_val)
                    
                elif move_type < 0.5 and len(A) > 30:
                    # Remove an element
                    old_val = random.choice(A)
                    A_new, new_reps, new_peak, new_score = remove_element_update(A, reps, old_val)
                    
                else:
                    # Swap: remove one, add one
                    old_val = random.choice(A)
                    candidates = [x for x in range(n + 1) if x not in A_set]
                    if not candidates:
                        continue
                    new_val = random.choice(candidates)
                    A_new, new_reps, new_peak, new_score = incremental_swap_update(A, reps, old_val, new_val)
                
                # Acceptance criterion
                if new_score < score:
                    A = A_new
                    reps = new_reps
                    peak = new_peak
                    score = new_score
                    A_set = set(A)
                    if peak < best_peak or (peak == best_peak and score < best_score):
                        best_peak = peak
                        best_score = score
                        best_A = list(A)
                else:
                    # Accept with probability exp(-delta/T)
                    delta = new_score - score
                    if delta < 0:
                        prob = 1.0
                    else:
                        prob = math.exp(-delta / T)
                    
                    if random.random() < prob:
                        A = A_new
                        reps = new_reps
                        peak = new_peak
                        score = new_score
                        A_set = set(A)
                
                T *= cooling
                if T < 0.01:
                    break
    
    # Phase 2: Local refinement with neutral moves
    if best_A is not None:
        if time.time() < deadline - 5:
            A = list(best_A)
            reps = compute_reps_full(A)
            peak = get_peak(reps)
            score = compute_score(reps)
            A_set = set(A)
            
            for attempt in range(5000):
                if time.time() > deadline:
                    break
                
                # Try swaps that maintain or improve score
                old_val = random.choice(A)
                candidates = [x for x in range(n + 1) if x not in A_set]
                if not candidates:
                    continue
                new_val = random.choice(candidates)
                
                A_new, new_reps, new_peak, new_score = incremental_swap_update(A, reps, old_val, new_val)
                
                if new_score <= score:
                    A = A_new
                    reps = new_reps
                    peak = new_peak
                    score = new_score
                    A_set = set(A)
                    if peak < best_peak or (peak == best_peak and score < best_score):
                        best_peak = peak
                        best_score = score
                        best_A = list(A)
                elif new_peak == peak and random.random() < 0.3:
                    # Accept neutral peak moves with some probability
                    A = A_new
                    reps = new_reps
                    score = new_score
                    A_set = set(A)
    
    # Phase 3: Try to reduce size while maintaining peak
    if best_A is not None and time.time() < deadline - 3:
        A = list(best_A)
        reps = compute_reps_full(A)
        peak = get_peak(reps)
        score = compute_score(reps)
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
            if len(A) <= 25:
                break
            
            # Try removing a random element
            idx = random.randint(0, len(A) - 1)
            A_trial = A[:idx] + A[idx+1:]
            new_reps = compute_reps_full(A_trial)
            new_peak = get_peak(new_reps)
            new_score = compute_score(new_reps)
            
            if new_peak <= peak:
                A = A_trial
                reps = new_reps
                peak = new_peak
                score = new_score
                if peak < best_peak or (peak == best_peak and score < best_score):
                    best_peak = peak
                    best_score = score
                    best_A = list(A)
    
    if best_A is None:
        # Fallback: a guaranteed valid set
        return list(range(64, 129))
    
    return best_A

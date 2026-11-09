import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128

    def evaluate(A):
        """
        Evaluates the set A using convolution for speed.
        Returns (peak, is_covered).
        peak: max multiplicity in [128, 256]
        is_covered: True if all sums in [128, 256] have count > 0
        """
        if not A:
            return float('inf'), False
            
        A_arr = np.array(list(A), dtype=np.int32)
        n = len(A_arr)
        
        # Use convolution to compute pair sums
        # Create indicator vector for A
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        indicator[A_arr] = 1
        
        # Convolution gives count of ordered pairs (a, b) with a + b = s
        # For unordered pairs with a <= b, we need to adjust
        conv = np.convolve(indicator, indicator, mode='full')
        
        # Vectorized count computation
        s_vals = np.arange(N_LOW, N_HIGH + 1, dtype=np.int32)
        conv_slice = conv[N_LOW:N_HIGH+1]
        
        # For even s: (conv[s] + indicator[s//2]) // 2
        # For odd s: conv[s] // 2
        even_mask = (s_vals % 2 == 0)
        counts = np.zeros(RANGE_SIZE, dtype=np.int32)
        
        # Odd sums: just conv[s] // 2
        counts[~even_mask] = conv_slice[~even_mask] // 2
        
        # Even sums: need to add indicator[s//2]
        if np.any(even_mask):
            even_s = s_vals[even_mask]
            mids = even_s // 2
            valid_mid = (mids >= 0) & (mids <= MAX_VAL)
            even_counts = conv_slice[even_mask] // 2
            even_counts[valid_mid] += indicator[mids[valid_mid]]
            # The formula is (conv[s] + indicator[s//2]) // 2
            # But conv[s] // 2 + indicator[s//2] is not the same as (conv[s] + indicator[s//2]) // 2
            # Let me redo: (conv[s] + indicator[s//2]) // 2
            # If conv[s] is even: conv[s]//2 + indicator[s//2]//2 = conv[s]//2 (since indicator is 0 or 1)
            # If conv[s] is odd: (conv[s]+1)//2 or (conv[s]+0)//2
            # Actually let's just compute it directly
            raw = conv_slice[even_mask] + indicator[mids]
            counts[even_mask] = raw // 2
        
        if not np.all(counts > 0):
            return float('inf'), False
            
        return int(np.max(counts)), True

    def get_counts(A):
        """Get the count array for evaluation."""
        if not A:
            return None
            
        A_arr = np.array(list(A), dtype=np.int32)
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        indicator[A_arr] = 1
        
        conv = np.convolve(indicator, indicator, mode='full')
        
        s_vals = np.arange(N_LOW, N_HIGH + 1, dtype=np.int32)
        conv_slice = conv[N_LOW:N_HIGH+1]
        
        even_mask = (s_vals % 2 == 0)
        counts = np.zeros(RANGE_SIZE, dtype=np.int32)
        
        counts[~even_mask] = conv_slice[~even_mask] // 2
        
        if np.any(even_mask):
            even_s = s_vals[even_mask]
            mids = even_s // 2
            valid_mid = (mids >= 0) & (mids <= MAX_VAL)
            raw = conv_slice[even_mask].copy()
            raw[valid_mid] += indicator[mids[valid_mid]]
            counts[even_mask] = raw // 2
        
        return counts

    best_peak = float('inf')
    best_A = None

    # Try several different initial sizes and restarts
    sizes_to_try = [35, 40, 45, 50, 55]
    random.shuffle(sizes_to_try)

    for target_size in sizes_to_try:
        if time.time() > deadline - 10:
            break
            
        # Initialize
        A_init = set()
        A_init.add(0)
        A_init.add(128)
        
        # Add random elements
        candidates = list(range(1, 128))
        random.shuffle(candidates)
        
        for c in candidates:
            if len(A_init) >= target_size:
                break
            A_init.add(c)
            
        current_A = A_init
        current_peak, is_covered = evaluate(current_A)
        
        if is_covered:
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = list(current_A)

        # Simulated Annealing
        T = 3.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        local_best_peak = current_peak if is_covered else float('inf')
        local_best_A = list(current_A) if is_covered else None
        stuck_count = 0
        
        while time.time() < deadline - 5 and T > min_T:
            # Choose move type with adaptive probabilities
            move_type = random.random()
            
            new_A = set(current_A)
            
            if move_type < 0.6:
                # Swap - try to swap elements that are in high-multiplicity sums
                if not new_A:
                    continue
                    
                # Get current counts to identify problematic elements
                counts = get_counts(new_A)
                if counts is not None and is_covered:
                    # Find sums with high multiplicity
                    high_mult_sums = np.where(counts > np.median(counts))[0]
                    if len(high_mult_sums) > 0:
                        # Pick a sum with high multiplicity and try to reduce it
                        target_sum = N_LOW + random.choice(high_mult_sums)
                        # Find elements that contribute to this sum
                        contributing = [a for a in new_A if (target_sum - a) in new_A and a <= target_sum - a]
                        if contributing:
                            to_remove = random.choice(contributing)
                            to_add = random.randint(0, 128)
                            if to_remove != to_add:
                                new_A.remove(to_remove)
                                new_A.add(to_add)
                            else:
                                continue
                        else:
                            to_remove = random.choice(list(new_A))
                            to_add = random.randint(0, 128)
                            if to_remove == to_add:
                                continue
                            new_A.remove(to_remove)
                            new_A.add(to_add)
                    else:
                        to_remove = random.choice(list(new_A))
                        to_add = random.randint(0, 128)
                        if to_remove == to_add:
                            continue
                        new_A.remove(to_remove)
                        new_A.add(to_add)
                else:
                    to_remove = random.choice(list(new_A))
                    to_add = random.randint(0, 128)
                    if to_remove == to_add:
                        continue
                    new_A.remove(to_remove)
                    new_A.add(to_add)
            elif move_type < 0.8:
                # Add
                if len(new_A) > 128:
                    continue
                to_add = random.randint(0, 128)
                if to_add in new_A:
                    continue
                new_A.add(to_add)
            else:
                # Remove
                if len(new_A) <= 2:
                    continue
                to_remove = random.choice(list(new_A))
                new_A.remove(to_remove)
                
            new_peak, is_covered_new = evaluate(new_A)
            
            if is_covered_new:
                if new_peak < local_best_peak:
                    local_best_peak = new_peak
                    local_best_A = list(new_A)
                    stuck_count = 0
                    
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_A = list(new_A)
                    stuck_count = 0
                
                # Acceptance criterion
                if is_covered:
                    delta = new_peak - current_peak
                    if delta < 0:
                        current_A = new_A
                        current_peak = new_peak
                    else:
                        # Prob of accepting worse - increase acceptance when stuck
                        if stuck_count > 100:
                            prob = np.exp(-delta / (T * 2))
                        else:
                            prob = np.exp(-delta / T)
                        if random.random() < prob:
                            current_A = new_A
                            current_peak = new_peak
                        else:
                            stuck_count += 1
                else:
                    # Was not covered, now covered: Accept
                    current_A = new_A
                    current_peak = new_peak
                    stuck_count = 0
            else:
                # Not covered
                if is_covered:
                    # Was covered, now not: Reject
                    stuck_count += 1
                else:
                    # Was not covered, still not covered:
                    # Check if we're making progress toward coverage
                    old_counts = get_counts(current_A)
                    new_counts = get_counts(new_A)
                    if old_counts is not None and new_counts is not None:
                        old_uncovered = np.sum(old_counts == 0)
                        new_uncovered = np.sum(new_counts == 0)
                        if new_uncovered < old_uncovered:
                            # Making progress toward coverage
                            current_A = new_A
                            stuck_count = 0
                        else:
                            stuck_count += 1
                    
            T *= cooling_rate
            
        # After SA for this size, update global best if local best is better
        if local_best_A is not None and local_best_peak < best_peak:
            best_peak = local_best_peak
            best_A = local_best_A

    # If no valid A was found, return a fallback
    if best_A is None:
        best_A = list(range(0, 129))

    # Final polish phase: try to reduce peak multiplicity further
    if time.time() < deadline - 2:
        polish_deadline = deadline - 1
        current_A = set(best_A)
        current_peak, is_covered = evaluate(current_A)
        
        if is_covered:
            # Try targeted swaps to reduce peak
            for polish_iter in range(200):
                if time.time() > polish_deadline:
                    break
                    
                counts = get_counts(current_A)
                if counts is None:
                    break
                    
                max_count = int(np.max(counts))
                if max_count <= 1:
                    break
                    
                # Find sums with maximum multiplicity
                peak_sums = np.where(counts == max_count)[0]
                target_sum_idx = random.choice(peak_sums)
                target_sum = N_LOW + target_sum_idx
                
                # Find elements contributing to this sum
                contributing = []
                A_list = list(current_A)
                for a in A_list:
                    b = target_sum - a
                    if a <= b and b in current_A:
                        contributing.append((a, b))
                
                if not contributing:
                    continue
                    
                # Try swapping one of the contributing elements
                a, b = random.choice(contributing)
                to_remove = a if random.random() < 0.5 else b
                to_add = random.randint(0, MAX_VAL)
                
                if to_add in current_A:
                    continue
                    
                new_A = set(current_A)
                new_A.remove(to_remove)
                new_A.add(to_add)
                
                new_peak, is_covered_new = evaluate(new_A)
                
                if is_covered_new and new_peak < current_peak:
                    current_A = new_A
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(current_A)
                elif is_covered_new and new_peak == current_peak:
                    # Accept neutral moves sometimes to explore
                    if random.random() < 0.3:
                        current_A = new_A

    return best_A

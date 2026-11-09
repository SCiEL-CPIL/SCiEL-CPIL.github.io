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
        
        # conv[s] = number of ordered pairs (a,b) in A×A with a+b=s
        # For s even: unordered pairs = (conv[s] + indicator[s//2]) // 2
        # For s odd: unordered pairs = conv[s] // 2
        
        counts = np.zeros(RANGE_SIZE, dtype=np.int32)
        for i in range(RANGE_SIZE):
            s = N_LOW + i
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= MAX_VAL:
                    counts[i] = (conv[s] + indicator[mid]) // 2
                else:
                    counts[i] = conv[s] // 2
            else:
                counts[i] = conv[s] // 2
                
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
        
        counts = np.zeros(RANGE_SIZE, dtype=np.int32)
        for i in range(RANGE_SIZE):
            s = N_LOW + i
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= MAX_VAL:
                    counts[i] = (conv[s] + indicator[mid]) // 2
                else:
                    counts[i] = conv[s] // 2
            else:
                counts[i] = conv[s] // 2
                
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

    return best_A

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)  # 129
    
    # Helper to calculate peak multiplicity and check coverage
    def evaluate(A_set):
        """Returns (peak_multiplicity, is_valid). A_set is a set of ints in [0,128]."""
        A = sorted(A_set)
        nA = len(A)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        
        for i in range(nA):
            ai = A[i]
            s_ii = ai + ai
            if min_sum <= s_ii <= max_sum:
                counts[s_ii] += 1
            for j in range(i + 1, nA):
                s_ij = ai + A[j]
                if min_sum <= s_ij <= max_sum:
                    counts[s_ij] += 1
        
        # Check coverage: every target sum must have at least one representation
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return (float('inf'), False)
        
        peak = int(np.max(target_counts))
        return (peak, True)
    
    def fast_evaluate(A_sorted):
        """Faster evaluation using numpy where possible. A_sorted is sorted list."""
        nA = len(A_sorted)
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        
        for i in range(nA):
            ai = A_sorted[i]
            # Vectorized for j > i
            if i + 1 < nA:
                sums = ai + np.array(A_sorted[i+1:])
                valid = (sums >= min_sum) & (sums <= max_sum)
                if np.any(valid):
                    np.add.at(counts, sums[valid].astype(int), 1)
            s_ii = ai + ai
            if min_sum <= s_ii <= max_sum:
                counts[s_ii] += 1
        
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return (float('inf'), False)
        
        peak = int(np.max(target_counts))
        return (peak, True)
    
    # Greedy initialization: start with a reasonable set
    # We need sums to cover [128, 256]. 
    # If A contains elements around 64, pairs sum to ~128.
    # Let's start with a set that guarantees coverage and then optimize.
    
    # Initial guess: random subset with good coverage properties
    # A good heuristic: include elements that help cover the extremes.
    # Min sum 128: need a+b=128. Max sum 256: need a+b=256 (so 128+128).
    # So 128 must be in A? Not necessarily, but 128+128=256. If 128 not in A, need two other elements summing to 256, impossible since max element is 128. So 128 MUST be in A.
    # Similarly, for sum 128: need a+b=128. If we have 0 and 128, that works.
    
    # Let's start with A = {0, 128} and add elements to cover all sums.
    # Actually, a better start: use a random set with ~40-60 elements.
    
    best_A = None
    best_peak = float('inf')
    
    # Generate initial candidate
    # Must include 128.
    # Let's try a greedy construction or random restart.
    
    random.seed(42)
    
    # Strategy: Local search on a set A.
    # Move: add an element, remove an element, or swap.
    
    def random_initial_set():
        A = set()
        A.add(128)  # Required for sum 256
        # Add random elements
        num_extra = random.randint(30, 60)
        candidates = list(range(0, 128))
        random.shuffle(candidates)
        for x in candidates[:num_extra]:
            A.add(x)
        return A
    
    # Try a few random initializations and pick the best
    for _ in range(5):
        A = random_initial_set()
        peak, valid = evaluate(A)
        if valid and peak < best_peak:
            best_peak = peak
            best_A = A.copy()
    
    if best_A is None:
        # Fallback: simple set
        best_A = set(range(0, 129, 4))
        best_A.add(128)
        peak, valid = evaluate(best_A)
        if not valid:
            # Ensure validity
            best_A = set(range(0, 129))
            best_peak, _ = evaluate(best_A)
        else:
            best_peak = peak
    
    # Local search
    A = best_A
    current_peak, _ = evaluate(A)
    
    no_improvement_count = 0
    max_no_improvement = 200
    
    while time.time() < deadline - 2.0:
        if no_improvement_count > max_no_improvement:
            break
        
        # Generate a random move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_A = A.copy()
        
        if move_type == 'add':
            # Add a random element
            x = random.randint(0, 128)
            if x not in new_A:
                new_A.add(x)
            else:
                continue
        elif move_type == 'remove':
            # Remove a random element (but keep 128 if possible, or handle invalidity)
            if len(new_A) > 2:  # Keep at least a few
                elements = list(new_A)
                # Avoid removing 128 if it's the only way to get 256? 
                # Actually, if we remove 128, we can't get 256 unless we have other elements summing to 256, which is impossible.
                # So 128 must stay if we want validity. Let's just not remove 128.
                removable = [x for x in elements if x != 128]
                if removable:
                    x = random.choice(removable)
                    new_A.remove(x)
                else:
                    continue
        elif move_type == 'swap':
            # Swap an element
            if len(new_A) > 2:
                elements = list(new_A)
                removable = [x for x in elements if x != 128]
                if removable:
                    x = random.choice(removable)
                    new_A.remove(x)
                    y = random.randint(0, 128)
                    if y not in new_A:
                        new_A.add(y)
                    else:
                        # Put it back
                        new_A.add(x)
                        continue
                else:
                    continue
        
        new_peak, valid = evaluate(new_A)
        
        if valid:
            if new_peak < current_peak:
                A = new_A
                current_peak = new_peak
                no_improvement_count = 0
                # Update global best
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = A.copy()
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
    
    # Return the best valid set found
    if best_A is None:
        # Should not happen
        return list(range(64, 129))
    
    return list(best_A)
# EVOLVE-BLOCK-END

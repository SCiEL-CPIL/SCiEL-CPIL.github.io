import numpy as np
import random
import time

def construct():
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = np.arange(target_lo, target_hi + 1)
    
    def get_cost(A_list):
        A = np.array(A_list)
        k = len(A)
        if k == 0:
            return 10000, 10000 # Peak, Uncovered
        S = A[:, None] + A[None, :]
        iu = np.triu_indices(k)
        S_tri = S[iu]
        mask = (S_tri >= target_lo) & (S_tri <= target_hi)
        S_filtered = S_tri[mask]
        if len(S_filtered) == 0:
            return 10000, 129
        counts = np.bincount(S_filtered, minlength=target_hi + 1)
        target_counts = counts[target_lo:target_hi + 1]
        peak = np.max(target_counts) if len(target_counts) > 0 else 0
        uncovered = np.sum(target_counts == 0)
        return peak, uncovered

    # 1. Greedy Initialization
    A = set()
    # Add 0 and 128 to help cover extremes? 
    # 128 can be 0+128, 1+127, etc.
    # 256 must be 128+128. So 128 MUST be in A.
    A.add(128)
    
    # We need to cover 128..256.
    # Greedy add elements to minimize uncovered count.
    uncovered = set(range(target_lo, target_hi + 1))
    
    def update_uncovered(A_set):
        reps = {s: 0 for s in range(target_lo, target_hi + 1)}
        elems = sorted(A_set)
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                if target_lo <= s <= target_hi:
                    reps[s] += 1
        return {s for s in range(target_lo, target_hi + 1) if reps[s] == 0}

    # Simple greedy to get a valid starting point
    candidates = list(range(0, 129))
    random.shuffle(candidates)
    start_time = time.time()
    
    # Ensure 128 is in A
    # Try to cover with few elements
    for _ in range(100):
        if not uncovered:
            break
        if time.time() - start_time > 5:
            break
        best_elem = None
        best_uncovered_count = len(uncovered)
        for e in candidates:
            if e in A:
                continue
            A.add(e)
            new_uncov = update_uncovered(A)
            if len(new_uncov) < best_uncovered_count:
                best_uncovered_count = len(new_uncov)
                best_elem = e
            A.remove(e)
        if best_elem is None:
            break
        A.add(best_elem)
        uncovered = update_uncovered(A)
        
    if uncovered:
        # Fallback: Add more elements until covered
        for e in range(0, 129):
            if e not in A:
                A.add(e)
                uncovered = update_uncovered(A)
                if not uncovered:
                    break
            if time.time() - start_time > 10:
                break
                # If still not covered, add everything? No, just break.
                # But we must return a valid set.
                # Let's force coverage by adding all remaining if needed?
                # Actually, {0..128} always covers.
                pass
        
    # Force valid if greedy failed
    if uncovered:
        A = set(range(0, 129))

    current_A = list(A)
    current_peak, current_uncovered = get_cost(current_A)
    
    # 2. Simulated Annealing
    if time.time() - start_time < 55:
        T = 10.0
        cooling = 0.99
        iterations = 0
        max_iters = 50000 # Vectorized, should be fast
        
        while T > 0.1 and iterations < max_iters:
            if time.time() - start_time > 55:
                break
            
            # Generate neighbor
            move_type = random.random()
            new_A = current_A[:]
            
            if move_type < 0.3:
                # Add
                if len(new_A) < 129:
                    e = random.randint(0, 128)
                    if e not in new_A:
                        new_A.append(e)
                    else:
                        continue
            elif move_type < 0.6:
                # Remove
                if len(new_A) > 1: # Keep at least 128? No, 128 is required for 256.
                    # Actually, 128 MUST be in A because 128+128=256.
                    # Are there other ways to make 256? Max element is 128.
                    # So only 128+128 works.
                    # So 128 cannot be removed.
                    removable = [x for x in new_A if x != 128]
                    if removable:
                        e = random.choice(removable)
                        new_A.remove(e)
                    else:
                        continue
            else:
                # Swap
                if len(new_A) > 1:
                    removable = [x for x in new_A if x != 128]
                    if removable:
                        e1 = random.choice(removable)
                        new_A.remove(e1)
                        e2 = random.randint(0, 128)
                        if e2 not in new_A:
                            new_A.append(e2)
                        else:
                            new_A.append(e1) # revert
                            continue
                    else:
                        continue
            
            new_peak, new_uncovered = get_cost(new_A)
            
            # Cost function
            # If uncovered, heavy penalty
            old_cost = current_peak + 100 * current_uncovered
            new_cost = new_peak + 100 * new_uncovered
            
            delta_cost = new_cost - old_cost
            
            if delta_cost < 0:
                current_A = new_A
                current_peak = new_peak
                current_uncovered = new_uncovered
            else:
                if random.random() < np.exp(-delta_cost / T):
                    current_A = new_A
                    current_peak = new_peak
                    current_uncovered = new_uncovered
            
            T *= cooling
            iterations += 1
    
    return sorted(current_A)

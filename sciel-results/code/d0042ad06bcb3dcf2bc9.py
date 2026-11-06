import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0
    max_n = 1000
    best_set = []
    best_size = 0
    
    # Precompute indices for speed
    s_range = np.arange(2, 2 * max_n + 1)
    
    def get_invalid_mask(chosen, counts):
        # counts is a numpy array of size 2001
        bad_sums = counts >= 2
        
        invalid_mask = np.zeros(max_n + 1, dtype=bool)
        
        # 1. 2c in bad_sums
        # c in 1..500, 2c in 2..1000
        # invalid_mask[c] = bad_sums[2c]
        c_indices = np.arange(1, max_n // 2 + 1)
        invalid_mask[c_indices] |= bad_sums[2 * c_indices]
        
        # 2. c + x in bad_sums for any x in chosen
        # c = s - x
        for x in chosen:
            # s in 2..2000
            # c = s - x
            c_vals = s_range - x
            valid_c = (c_vals >= 1) & (c_vals <= max_n)
            if np.any(valid_c):
                # Map valid c_vals to indices
                idx = c_vals[valid_c]
                # If any corresponding s has bad_sums True, mark c as invalid
                invalid_mask[idx] |= bad_sums[s_range[valid_c]]
                
        return invalid_mask

    def run_local_search():
        chosen = []
        counts = np.zeros(2 * max_n + 1, dtype=np.int32)
        
        # Randomized Greedy Init
        candidates = list(range(1, max_n + 1))
        random.shuffle(candidates)
        
        for c in candidates:
            # Fast check: 
            # We can just use the invalid mask logic, but for init, 
            # we can just check directly to avoid overhead if set is small.
            # Or just use the mask.
            # For init, let's just do direct check for simplicity, it's O(N^2) total which is fine.
            if counts[2 * c] < 2:
                valid = True
                for x in chosen:
                    if counts[c + x] >= 2:
                        valid = False
                        break
                if valid:
                    counts[2 * c] += 1
                    chosen.append(c)
                    for x in chosen[:-1]:
                        counts[c + x] += 1
        
        current_size = len(chosen)
        
        stuck_counter = 0
        while time.time() - start_time < deadline - 1.0:
            # Calculate invalid mask
            invalid_mask = get_invalid_mask(chosen, counts)
            
            # Find candidates
            # Exclude already chosen
            chosen_set = set(chosen)
            candidates = [i for i in range(1, max_n + 1) if not invalid_mask[i] and i not in chosen_set]
            
            if candidates:
                # Pick a candidate
                # Strategy: Pick the one that minimizes future conflicts? 
                # For now, random or smallest.
                c = random.choice(candidates)
                
                # Add it
                counts[2 * c] += 1
                chosen.append(c)
                for x in chosen[:-1]:
                    counts[c + x] += 1
                current_size += 1
                stuck_counter = 0
            else:
                # Stuck, try swap
                if not chosen:
                    break
                
                a = random.choice(chosen)
                # Find candidates for b
                # We need to check validity of A \ {a} + {b}
                # This is hard to vectorize easily with the current mask.
                # Let's just pick a random b from 1..1000 not in chosen
                # and check if swap is valid.
                
                # To speed up, we can pick b from a list of "likely" candidates.
                # Or just random.
                b = random.randint(1, max_n)
                while b in chosen_set:
                    b = random.randint(1, max_n)
                
                # Check swap
                # Temporarily remove a
                counts[2 * a] -= 1
                chosen.remove(a)
                for x in chosen:
                    counts[a + x] -= 1
                
                # Check if b can be added
                valid_b = True
                if counts[2 * b] >= 2:
                    valid_b = False
                else:
                    for x in chosen:
                        if counts[b + x] >= 2:
                            valid_b = False
                            break
                
                if valid_b:
                    # Perform swap
                    counts[2 * b] += 1
                    chosen.append(b)
                    for x in chosen[:-1]:
                        counts[b + x] += 1
                    stuck_counter = 0
                else:
                    # Revert
                    counts[2 * a] += 1
                    chosen.append(a)
                    for x in chosen:
                        if x != a: # a is not in chosen yet
                            counts[a + x] += 1
                    stuck_counter += 1
                    
                    if stuck_counter > 100:
                        break
                        
        return chosen, current_size

    while time.time() - start_time < deadline:
        current_set, current_size = run_local_search()
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            
    return best_set

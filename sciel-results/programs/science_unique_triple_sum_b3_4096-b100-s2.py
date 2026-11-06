import time
import random
import numpy as np

def construct():
    deadline = time.time() + 50.0
    best_set = []
    n_max = 4096
    max_sum = 3 * n_max
    
    # Helper to check and add x to a set represented by list and sum array
    def try_add(x, selected, sums_arr):
        # 1. (x, x, x)
        s = 3 * x
        if sums_arr[s]:
            return False
        new_sums = [s]
        
        # 2. (y, x, x) -> y + 2x
        # 3. (y, y, x) -> 2y + x
        for y in selected:
            s1 = y + 2 * x
            if sums_arr[s1]:
                return False
            new_sums.append(s1)
            
            s2 = 2 * y + x
            if sums_arr[s2]:
                return False
            new_sums.append(s2)
            
        # 4. (y, z, x) with y < z in selected
        n_sel = len(selected)
        for i in range(n_sel):
            y = selected[i]
            for j in range(i + 1, n_sel):
                z = selected[j]
                s = y + z + x
                if sums_arr[s]:
                    return False
                new_sums.append(s)
                
        # If all good, commit
        for s in new_sums:
            sums_arr[s] = True
        selected.append(x)
        return True

    def greedy_pass(order):
        selected = []
        sums_arr = np.zeros(max_sum + 1, dtype=bool)
        for x in order:
            if time.time() > deadline:
                break
            if try_add(x, selected, sums_arr):
                if len(selected) > len(best_set):
                    # Optional: deep copy if we want to keep intermediate bests?
                    # No, just update at end of pass or if significantly better.
                    # Actually, we should update best_set if this pass beats it.
                    # But we don't know the final size until the pass is done.
                    # So we just return the selected set at the end.
                    pass
        return selected

    # Strategy 1: Random Greedy
    # Strategy 2: Skeleton Greedy
    
    # Skeleton: Powers of 4
    skeleton = [4**i for i in range(7) if 4**i <= n_max]
    # skeleton is [1, 4, 16, 64, 256, 1024, 4096]
    
    # We can also try other skeletons, e.g., scaled powers of 2 that are valid?
    # But we know powers of 2 are invalid.
    # Let's just use random and skeleton.
    
    while time.time() < deadline:
        # Random Pass
        candidates = list(range(1, n_max + 1))
        random.shuffle(candidates)
        res = greedy_pass(candidates)
        if len(res) > len(best_set):
            best_set = res
            
        if time.time() > deadline:
            break
        
        # Skeleton Pass
        # Start with skeleton, then try to add others
        # We need to initialize sums_arr with skeleton sums
        selected = list(skeleton)
        sums_arr = np.zeros(max_sum + 1, dtype=bool)
        
        # Add skeleton sums
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                for k in range(j, len(selected)):
                    sums_arr[selected[i] + selected[j] + selected[k]] = True
        
        # Candidates to try: all numbers not in skeleton
        # Shuffle them
        remaining = [x for x in range(1, n_max + 1) if x not in set(skeleton)]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline:
                break
            if try_add(x, selected, sums_arr):
                pass
                
        if len(selected) > len(best_set):
            best_set = selected
            
        # Occasionally try a different skeleton?
        # Maybe powers of 3?
        # 1, 3, 9, 27, 81, 243, 729, 2187, 6561 (too big)
        # Check if powers of 3 are valid B3[1]?
        # 3^a + 3^b + 3^c. Base 3 digits are 1.
        # 1+1+1 = 3 = 10_3. Carry!
        # So powers of 3 are NOT valid.
        # Powers of 4 are valid because 1+1+1 = 3 < 4. No carry.
        
        # What about a mix?
        # Let's just rely on randomization and the one good skeleton.
        
        # To add variety, we can randomize the skeleton slightly?
        # No, keep it simple.
        
    return best_set

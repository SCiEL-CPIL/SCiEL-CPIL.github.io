import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave buffer for 90s limit
    target_size = 40
    
    def verify(s):
        """Verify that the set has unique triple sums."""
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 4096 for x in s):
            return False
        sums = [sum(t) for t in combinations_with_replacement(s, 3)]
        return len(sums) == len(set(sums))

    def run_greedy(order, time_budget, start_time_ref, initial_selected=None):
        """
        Greedy construction.
        order: iterable of candidates
        time_budget: seconds allowed
        initial_selected: list of already selected elements
        """
        if initial_selected is None:
            initial_selected = []
            
        selected = list(initial_selected)
        if len(selected) >= target_size:
            return selected
            
        # Initialize sum mask (or set)
        # Using a set of integers is fast enough for ~10k sums
        used_sums = set()
        if len(selected) > 0:
            sel_arr = np.array(selected, dtype=np.int32)
            for t in combinations_with_replacement(sel_arr, 3):
                used_sums.add(int(t[0] + t[1] + t[2]))
        
        selected_arr = np.array(selected, dtype=np.int32)
        
        for x in order:
            if time.time() - start_time_ref > time_budget:
                break
            if len(selected) >= target_size:
                break
            
            x_int = int(x)
            
            # Generate all new sums involving x
            new_sums = []
            
            # 1. (x, x, x)
            s_0 = 3 * x_int
            new_sums.append(s_0)
            
            if len(selected_arr) > 0:
                # 2. (x, x, y)
                sums_xx_y = 2 * x_int + selected_arr
                new_sums.extend(sums_xx_y.tolist())
                
                # 3. (x, y, y)
                sums_x_yy = x_int + 2 * selected_arr
                new_sums.extend(sums_x_yy.tolist())
                
                # 4. (x, y, z) for y < z in selected
                n = len(selected_arr)
                for i in range(n):
                    for j in range(i + 1, n):
                        s = x_int + selected_arr[i] + selected_arr[j]
                        new_sums.append(s)
            
            # Check for collisions
            # Convert to numpy array for fast membership check against a set?
            # Actually, checking a list of ~800 items against a set of ~10k items is fast.
            # But converting to set and intersecting might be faster?
            # Or just loop?
            # np.intersect1d is slow.
            # Let's use a set of new_sums and check intersection with used_sums.
            new_sums_set = set(new_sums)
            if new_sums_set & used_sums:
                continue
            
            # Accept x
            used_sums.update(new_sums_set)
            selected.append(x_int)
            selected_arr = np.append(selected_arr, x_int)
            
            if len(selected) >= target_size:
                break
        
        return selected

    best_set = []
    
    # Strategy 1: Deterministic Greedy (1 to 4096)
    s1 = run_greedy(range(1, 4097), time_limit - (time.time() - start_time), start_time)
    if len(s1) > len(best_set):
        best_set = s1
    if len(best_set) >= target_size:
        return best_set

    # Strategy 2: Deterministic Greedy (4096 to 1)
    if time.time() - start_time < time_limit - 10:
        s2 = run_greedy(range(4096, 0, -1), time_limit - (time.time() - start_time), start_time)
        if len(s2) > len(best_set):
            best_set = s2
        if len(best_set) >= target_size:
            return best_set

    # Strategy 3: Randomized Greedy
    random.seed(42)
    for trial in range(200): # Increased trials
        if time.time() - start_time > time_limit - 5:
            break
        
        # Random order
        order = list(range(1, 4097))
        random.shuffle(order)
        
        s = run_greedy(order, time_limit - (time.time() - start_time), start_time)
        if len(s) > len(best_set):
            best_set = s
            if len(best_set) >= target_size:
                return best_set

    # Strategy 4: Local Search / Repair on best_set
    # If we have 39, try to swap one out to fit one in?
    # This is complex. Given time, we rely on greedy.
    
    if not verify(best_set):
        # Fallback: Powers of 4
        best_set = [1, 4, 16, 64, 256, 1024, 4096]
        
    return best_set

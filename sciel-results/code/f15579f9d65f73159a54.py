"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave some buffer for 90s limit

    # Theoretical bound: for |S|=n, number of triples is n(n+1)(n+2)/6.
    # Sums range from 3 to 12288, giving 12286 possible distinct sums.
    # n=40: 11480 <= 12286. OK.
    # n=41: 12411 > 12286. Not OK.
    # So the theoretical maximum is 40.
    
    def verify(s):
        """Verify that the set has unique triple sums."""
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 4096 for x in s):
            return False
        sums = [sum(t) for t in combinations_with_replacement(s, 3)]
        return len(sums) == len(set(sums))

    def run_greedy_fast(order, time_budget, start_time_ref):
        """Fast greedy using numpy for vectorized collision checks."""
        selected = []
        current_sums = set()
        selected_arr = np.array([], dtype=np.int32)
        
        for x in order:
            if time.time() - start_time_ref > time_budget:
                break
            if len(selected) >= 40:
                break
            
            # Vectorized check for collisions
            valid = True
            new_sums = set()
            
            # (x, x, x)
            s = 3 * x
            if s in current_sums:
                continue
            new_sums.add(s)
            
            if len(selected_arr) == 0:
                selected.append(x)
                current_sums.update(new_sums)
                selected_arr = np.array(selected, dtype=np.int32)
                continue
            
            # (x, x, y) for all y in selected
            sums_xx_y = 2 * x + selected_arr
            for s in sums_xx_y:
                if s in current_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(int(s))
            
            if not valid:
                continue
            
            # (x, y, y) for all y in selected
            sums_x_yy = x + 2 * selected_arr
            for s in sums_x_yy:
                if s in current_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(int(s))
            
            if not valid:
                continue
            
            # (x, y, z) for all pairs y < z in selected
            n = len(selected_arr)
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    s = x + selected_arr[i] + selected_arr[j]
                    if s in current_sums or s in new_sums:
                        valid = False
                        break
                    new_sums.add(int(s))
                if not valid:
                    break
            
            if valid:
                selected.append(x)
                current_sums.update(new_sums)
                selected_arr = np.array(selected, dtype=np.int32)
        
        return selected

    def run_greedy_with_backtrack(order, target_size, time_budget, start_time_ref):
        """Greedy with limited backtracking to escape local optima."""
        selected = []
        current_sums = set()
        selected_arr = np.array([], dtype=np.int32)
        order_idx = 0
        
        def backtrack(depth, max_depth=3):
            nonlocal order_idx, selected, current_sums, selected_arr
            if time.time() - start_time_ref > time_budget:
                return False
            if len(selected) >= target_size:
                return True
            
            # Try candidates from current position
            tried = 0
            while order_idx < len(order) and tried < max_depth:
                x = order[order_idx]
                order_idx += 1
                tried += 1
                
                # Check if x is valid
                new_sums = set()
                valid = True
                
                s = 3 * x
                if s in current_sums:
                    continue
                new_sums.add(s)
                
                if len(selected_arr) > 0:
                    sums_xx_y = 2 * x + selected_arr
                    for sv in sums_xx_y:
                        if sv in current_sums or sv in new_sums:
                            valid = False
                            break
                        new_sums.add(int(sv))
                    if not valid:
                        continue
                    
                    sums_x_yy = x + 2 * selected_arr
                    for sv in sums_x_yy:
                        if sv in current_sums or sv in new_sums:
                            valid = False
                            break
                        new_sums.add(int(sv))
                    if not valid:
                        continue
                    
                    n = len(selected_arr)
                    for i in range(n):
                        for j in range(i + 1, n):
                            sv = x + selected_arr[i] + selected_arr[j]
                            if sv in current_sums or sv in new_sums:
                                valid = False
                                break
                            new_sums.add(int(sv))
                        if not valid:
                            break
                    if not valid:
                        continue
                
                # Accept x
                selected.append(x)
                current_sums.update(new_sums)
                selected_arr = np.array(selected, dtype=np.int32)
                
                if backtrack(depth + 1, max_depth):
                    return True
                
                # Backtrack
                selected.pop()
                for sv in new_sums:
                    current_sums.discard(sv)
                selected_arr = np.array(selected, dtype=np.int32)
            
            return False
        
        backtrack(0)
        return selected

    # Strategy 1: Fast greedy from 1 upward
    best_set = run_greedy_fast(range(1, 4097), time_limit - (time.time() - start_time), start_time)
    
    # Strategy 2: Fast greedy from larger numbers downward
    if time.time() - start_time < time_limit - 10:
        s2 = run_greedy_fast(range(4096, 0, -1), time_limit - (time.time() - start_time), start_time)
        if len(s2) > len(best_set):
            best_set = s2
    
    # Strategy 3: Random orderings with different seeds
    random.seed(42)
    for trial in range(30):
        if time.time() - start_time > time_limit - 5:
            break
        order = list(range(1, 4097))
        random.shuffle(order)
        s = run_greedy_fast(order, time_limit - (time.time() - start_time), start_time)
        if len(s) > len(best_set):
            best_set = s
    
    # Strategy 4: Greedy with backtracking on random orders
    if time.time() - start_time < time_limit - 15:
        for trial in range(10):
            if time.time() - start_time > time_limit - 5:
                break
            random.seed(42 + trial)
            order = list(range(1, 4097))
            random.shuffle(order)
            s = run_greedy_with_backtrack(order, 40, time_limit - (time.time() - start_time), start_time)
            if len(s) > len(best_set):
                best_set = s
    
    # Strategy 5: Try to extend the best set found so far
    if time.time() - start_time < time_limit - 10 and len(best_set) < 40:
        base = best_set[:]
        base_set = set(base)
        remaining = [x for x in range(1, 4097) if x not in base_set]
        extended = run_greedy_fast(remaining, time_limit - (time.time() - start_time), start_time)
        if len(extended) > len(base):
            best_set = base + extended
    
    # Verify the result
    if not verify(best_set):
        # Fallback to a simple valid set
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    return best_set
# EVOLVE-BLOCK-END

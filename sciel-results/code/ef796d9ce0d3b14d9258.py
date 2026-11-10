import time
import numpy as np
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit (evaluator kills at 90s, but we want to be safe)
    
    max_val = 256
    max_size = 30
    max_sum = max_size * max_val + 1  # 7681
    
    # Initial seed set known to be valid
    # {31, 32, 34} -> sums: 31, 32, 34, 63, 65, 66, 97
    # Check divisibility:
    # 31: 63(no), 65(no), 66(no), 97(no), 32(no), 34(no)
    # 32: 63(no), 65(no), 66(no), 97(no), 31(no), 34(no)
    # 34: 63(no), 65(no), 66(no), 97(no), 31(no), 32(no)
    # 63: 65(no), 66(no), 97(no), 31(no), 32(no), 34(no)
    # 65: 66(no), 97(no), ...
    # 66: 97(no), ...
    # 97: ...
    # Valid.
    
    best_set = [31, 32, 34]
    
    def is_valid_set(arr):
        """
        Verify a set is valid.
        Returns True if valid, False otherwise.
        """
        if len(arr) == 0:
            return True
        if len(arr) > max_size:
            return False
        if any(x < 1 or x > max_val for x in arr):
            return False
        if len(set(arr)) != len(arr):
            return False
            
        sums = np.zeros(max_sum + 1, dtype=bool)
        sums[0] = True
        for x in arr:
            # Shift and OR
            # temp_sums = sums | (sums shifted by x)
            # Since we are verifying, we can just build the set of sums.
            # For speed, use the boolean array method.
            new_sums = sums.copy()
            if x <= max_sum:
                new_sums[x:] = new_sums[x:] | sums[:max_sum + 1 - x]
            sums = new_sums
            
        # Check divisibility
        # Get indices of True values (excluding 0)
        active_indices = np.nonzero(sums)[0]
        active_indices = active_indices[active_indices > 0]
        
        if len(active_indices) == 0:
            return True
            
        # Sort them (they are already sorted from np.nonzero)
        # For each i, check if any multiple is in the set.
        # To do this fast, we can iterate i and check multiples.
        # Since max_sum is small, this is fast.
        for i in active_indices:
            if i > max_sum // 2:
                continue
            # Check multiples 2*i, 3*i, ...
            # If any is in sums, invalid.
            # We can check sums[2*i : max_sum+1 : i].any()
            if np.any(sums[2*i : max_sum + 1 : i]):
                return False
        return True

    def greedy_extend(current_set, candidates, deadline):
        """
        Greedily extend current_set with candidates.
        Returns the extended set.
        """
        current_set = list(current_set)
        if not current_set:
            return current_set
            
        # Initialize sums array from current_set
        sums = np.zeros(max_sum + 1, dtype=bool)
        sums[0] = True
        for x in current_set:
            new_sums = sums.copy()
            if x <= max_sum:
                new_sums[x:] = new_sums[x:] | sums[:max_sum + 1 - x]
            sums = new_sums
            
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current_set) >= max_size:
                break
            
            # Check if x can be added
            temp_sums = sums.copy()
            if x <= max_sum:
                temp_sums[x:] = temp_sums[x:] | sums[:max_sum + 1 - x]
                
            # Check validity
            valid = True
            # Only need to check new sums?
            # No, we need to ensure the whole set is valid.
            # But old set was valid.
            # So we only need to check:
            # 1. New sums vs Old sums
            # 2. New sums vs New sums
            # 
            # Actually, the full check `np.any(sums[2*i...])` on temp_sums is fast enough (~100k ops).
            # Let's just do the full check on temp_sums to be safe and simple.
            # Optimization: We can restrict the check to i that are in temp_sums.
            # active = np.nonzero(temp_sums)[0]
            # active = active[active > 0]
            # for i in active: ...
            
            # Even faster:
            # We know old_sums was valid.
            # New sums are N = {s+x | s in old_sums}.
            # We need to check:
            # - n1 divides n2 for n1, n2 in N
            # - n divides o or o divides n for n in N, o in old_sums
            
            # The full check is simpler and likely fast enough given the small max_sum.
            # Let's stick to the full check but optimized loop.
            
            active = np.nonzero(temp_sums)[0]
            active = active[active > 0]
            
            for i in active:
                if i > max_sum // 2:
                    continue
                if np.any(temp_sums[2*i : max_sum + 1 : i]):
                    valid = False
                    break
                    
            if valid:
                current_set.append(x)
                sums = temp_sums
                
        return current_set

    # Strategy 1: Systematic Greedy
    # Try different orders of candidates
    
    # Order 1: Descending
    candidates_desc = list(range(256, 0, -1))
    res = greedy_extend([], candidates_desc, deadline)
    if len(res) > len(best_set):
        best_set = res
        
    # Order 2: Ascending
    candidates_asc = list(range(1, 257))
    res = greedy_extend([], candidates_asc, deadline)
    if len(res) > len(best_set):
        best_set = res
        
    # Order 3: Randomized
    # Use multiple random restarts
    
    # To maximize chances, we can try to start from the current best_set and extend?
    # Or start from scratch with random orders.
    
    # Let's try random orders starting from scratch.
    for trial in range(1000):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        res = greedy_extend([], candidates, deadline)
        if len(res) > len(best_set):
            best_set = res
            
    # Try extending the best_set found so far with remaining candidates?
    # This might help if the greedy path got stuck early but could have taken a different path later?
    # No, greedy is deterministic given the order.
    # But maybe we can try to "repair" or "improve" the best set?
    # The problem is "maximize set size".
    
    # Another idea: Local Search
    # Take best_set, try to swap elements or add/remove to increase size?
    # Since we are maximizing size, adding is the goal.
    # If we can't add, maybe removing one allows adding two?
    # This is complex.
    
    # Let's try a few more systematic orders.
    # Prime-heavy?
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
    primes = [p for p in primes if p <= 256]
    res = greedy_extend([], primes, deadline)
    if len(res) > len(best_set):
        best_set = res
        
    # Composites?
    composites = [x for x in range(1, 257) if x not in primes and x > 1]
    res = greedy_extend([], composites, deadline)
    if len(res) > len(best_set):
        best_set = res
        
    # Mixed random
    for trial in range(500):
        if time.time() > deadline:
            break
        random.seed(trial + 10000)
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        res = greedy_extend([], candidates, deadline)
        if len(res) > len(best_set):
            best_set = res
            
    # Final validation
    if not is_valid_set(best_set):
        # Fallback to known good
        best_set = [31, 32, 34]
        
    # Ensure types are correct
    return [int(x) for x in best_set]

# Test locally?
# if __name__ == "__main__":
#     print(construct())

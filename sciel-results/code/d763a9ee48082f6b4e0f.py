import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    def check_addition(S, x, sums_set, sums_array):
        """
        Check if x can be added to S while maintaining unique triple sums.
        S is a sorted list of current elements.
        sums_set is a set of all current triple sums.
        sums_array is a sorted numpy array of all current triple sums.
        Returns (True, new_sums_list, new_sums_array) if valid, (False, None, None) otherwise.
        """
        n = len(S)
        if n == 0:
            return True, [3*x], np.array([3*x])
        
        S_arr = np.array(S)
        
        # 1. (x, x, x)
        s1 = 3 * x
        
        # 2. (x, x, s) for s in S
        s2s = 2 * x + S_arr
        
        # 3. (x, s, t) for s <= t in S
        # We need sums x + s_i + s_j for i <= j
        # np.add.outer(S_arr, S_arr) gives all s_i + s_j
        # We only need i <= j, but checking all is fine if we handle uniqueness correctly?
        # No, if we check all, we might find a collision that doesn't exist in the unordered sense?
        # Wait, the sum s_i + s_j is the same as s_j + s_i.
        # The set of sums {x + s_i + s_j | i <= j} is a subset of {x + s_i + s_j | i, j}.
        # If we generate all i,j, we get duplicates.
        # If we use np.unique on the new sums, duplicates are removed.
        # So we can just generate all pairs.
        
        s3s = x + np.add.outer(S_arr, S_arr)
        
        # Combine all new sums
        # Note: s1 is a scalar, s2s is 1D, s3s is 2D (flattened)
        new_sums = np.concatenate((np.array([s1]), s2s, s3s.flatten()))
        
        # Check for internal collisions in new_sums
        # If there are duplicates in new_sums, it's a collision
        if len(np.unique(new_sums)) != len(new_sums):
            return False, None, None
            
        # Check for collisions with existing sums_array
        if len(sums_array) > 0:
            idx = np.searchsorted(sums_array, new_sums)
            # Check if the index is within bounds and matches
            # Need to be careful with bounds
            valid_idx = idx < len(sums_array)
            if np.any(valid_idx):
                # Check matches at valid indices
                matches = sums_array[idx[valid_idx]] == new_sums[valid_idx]
                if np.any(matches):
                    return False, None, None
            
        # Valid
        # Update sums_array
        new_sums_array = np.concatenate((sums_array, new_sums))
        new_sums_array.sort()
        
        return True, new_sums, new_sums_array

    def greedy_search(order):
        """
        Greedily add elements from 'order' to the set.
        Returns (set_of_elements, set_of_sums, array_of_sums).
        """
        S = []
        sums_set = set()
        sums_array = np.array([], dtype=np.int64)
        
        for x in order:
            if time.time() > deadline:
                break
            valid, new_sums, new_sums_array = check_addition(S, x, sums_set, sums_array)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
                sums_array = new_sums_array
                # S is not strictly needed to be sorted for the logic if we use np.add.outer
                # but keeping it sorted might help with reproducibility or other heuristics
                # S.sort() 
                
                if len(S) >= 80: # Max allowed
                    break
                    
        return S, sums_set, sums_array

    best_S = []
    best_sums = set()
    best_sums_array = np.array([], dtype=np.int64)
    
    # Strategy 1: Standard Greedy (1, 2, 3...)
    # This is the baseline.
    S1, sums1, arr1 = greedy_search(range(1, 4097))
    if len(S1) > len(best_S):
        best_S = S1
        best_sums = sums1
        best_sums_array = arr1
        
    # Strategy 2: Randomized Greedy
    num_restarts = 5000 # Will be limited by time, now we can afford more
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random shuffle of candidates
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        S, sums_set, sums_array = greedy_search(candidates)
        
        if len(S) > len(best_S):
            best_S = S
            best_sums = sums_set
            best_sums_array = sums_array
            
        # Early stop if we find a very large set?
        # Theoretical max is around 40-42.
        if len(best_S) >= 40:
            break
                
    return best_S

# Quick test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)

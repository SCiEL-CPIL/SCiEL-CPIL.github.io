import time
import random
import numpy as np
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    best_size = 0
    best_set = set()
    
    def greedy_with_cost(seed=0):
        random.seed(seed)
        S_list = []  # Sorted list of elements in S
        S_set = set() # For O(1) membership
        counts = np.zeros(2 * N + 1, dtype=np.int8)
        
        for _ in range(N):
            # Get candidates
            candidates = np.array([x for x in range(1, N + 1) if x not in S_set], dtype=np.int32)
            
            if len(candidates) == 0:
                break
            
            S_arr = np.array(S_list, dtype=np.int32)
            
            # Matrix of sums: candidates (M,1) + S (1,K) -> (M,K)
            sums_matrix = candidates[:, None] + S_arr[None, :]
            
            # Get counts for these sums
            c_sums = counts[sums_matrix]
            
            # Validity check for x+y
            valid_sums = np.all(c_sums < 2, axis=1)
            
            # Validity check for 2x
            c_2x = counts[2 * candidates]
            valid_2x = c_2x < 2
            
            is_valid = valid_sums & valid_2x
            
            if not np.any(is_valid):
                break
            
            # Calculate costs
            costs = np.sum(c_sums == 1, axis=1) + (c_2x == 1).astype(np.int8)
            
            # Mask invalid costs with infinity
            costs[~is_valid] = np.inf
            
            # Find min cost
            min_cost = np.min(costs)
            
            # Get all candidates with min cost
            best_indices = np.where(costs == min_cost)[0]
            
            # Random choice among best
            choice_idx = random.choice(best_indices)
            x = candidates[choice_idx]
            
            # Update S
            S_set.add(x)
            bisect.insort(S_list, x)
            
            # Update counts
            counts[2 * x] += 1
            if S_arr.size > 0:
                new_sums = x + S_arr
                counts[new_sums] += 1
        
        return S_set

    # Run multiple restarts until deadline
    i = 0
    while time.time() < deadline - 5:
        S = greedy_with_cost(seed=i)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        i += 1
            
    return list(best_set)

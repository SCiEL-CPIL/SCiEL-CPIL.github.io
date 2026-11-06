# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    best_set = []
    best_size = 0

    def _greedy_numpy(order, counts_arr):
        """Greedy construction using numpy array for fast sum counting."""
        chosen = []
        # counts_arr is a numpy array of size 2001 (sums range from 2 to 2000)
        for x in order:
            # Check 2*x
            if counts_arr[2 * x] >= 2:
                continue
            # Check x + y for all y in chosen
            valid = True
            for y in chosen:
                s = x + y
                if counts_arr[s] >= 2:
                    valid = False
                    break
            if valid:
                chosen.append(x)
                counts_arr[2 * x] += 1
                for y in chosen[:-1]:
                    s = x + y
                    counts_arr[s] += 1
        return chosen

    def _local_search_improve(chosen, time_limit):
        """Try to improve the set by swapping elements or adding new ones."""
        chosen = list(chosen)
        counts_arr = np.zeros(2001, dtype=np.int32)
        
        # Rebuild counts
        for i, a in enumerate(chosen):
            counts_arr[2 * a] += 1
            for b in chosen[i+1:]:
                counts_arr[a + b] += 1
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(chosen)
            
            # Try adding a new element
            for x in range(1, 1001):
                if x in chosen:
                    continue
                # Check if x can be added
                if counts_arr[2 * x] >= 2:
                    continue
                can_add = True
                for y in chosen:
                    if counts_arr[x + y] >= 2:
                        can_add = False
                        break
                if can_add:
                    chosen.append(x)
                    counts_arr[2 * x] += 1
                    for y in chosen[:-1]:
                        counts_arr[x + y] += 1
                    improved = True
                    break
            
            if not improved:
                # Try swapping: remove one element, add another
                for i in range(n):
                    old = chosen[i]
                    # Remove old from counts
                    counts_arr[2 * old] -= 1
                    for j, y in enumerate(chosen):
                        if j != i:
                            counts_arr[old + y] -= 1
                    
                    # Try to add a new element
                    for x in range(1, 1001):
                        if x == old or x in chosen:
                            continue
                        if counts_arr[2 * x] >= 2:
                            continue
                        can_add = True
                        for y in chosen:
                            if y != old and counts_arr[x + y] >= 2:
                                can_add = False
                                break
                        if can_add:
                            chosen[i] = x
                            counts_arr[2 * x] += 1
                            for y in chosen:
                                if y != x:
                                    counts_arr[x + y] += 1
                            improved = True
                            break
                    
                    if not improved:
                        # Restore old
                        counts_arr[2 * old] += 1
                        for y in chosen:
                            if y != old:
                                counts_arr[old + y] += 1
                    else:
                        break
        
        return chosen

    # Strategy 1: Forward greedy (small to large)
    order1 = list(range(1, 1001))
    counts1 = np.zeros(2001, dtype=np.int32)
    set1 = _greedy_numpy(order1, counts1)
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1

    # Strategy 2: Backward greedy (large to small)
    order2 = list(range(1000, 0, -1))
    counts2 = np.zeros(2001, dtype=np.int32)
    set2 = _greedy_numpy(order2, counts2)
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2

    # Strategy 3: Random restarts with local search
    random.seed(42)
    while time.time() < deadline - 30:  # Leave 30s for final local search
        # Random permutation with bias
        order = list(range(1, 1001))
        random.shuffle(order)
        
        counts = np.zeros(2001, dtype=np.int32)
        current_set = _greedy_numpy(order, counts)
        
        # Apply local search if time permits
        if time.time() < deadline - 10:
            local_time_limit = min(time.time() + 5, deadline - 5)
            current_set = _local_search_improve(current_set, local_time_limit)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # Final local search on best set
    if time.time() < deadline - 5:
        best_set = _local_search_improve(best_set, deadline - 2)
        best_size = len(best_set)

    return best_set
# EVOLVE-BLOCK-END

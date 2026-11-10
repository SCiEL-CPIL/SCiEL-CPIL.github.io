import time
import random
import numpy as np

def _try_greedy_fast(order, deadline):
    """Fast greedy using NumPy for counts."""
    chosen = []
    chosen_arr = np.array([], dtype=np.int32)
    counts = np.zeros(2001, dtype=np.int8)
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Check validity
        if counts[2 * x] >= 2:
            continue
            
        if len(chosen) > 0:
            sums = x + chosen_arr
            if np.any(counts[sums] >= 2):
                continue
        
        # Valid
        chosen.append(x)
        chosen_arr = np.append(chosen_arr, x)
        counts[2 * x] += 1
        if len(chosen) > 1:
            counts[x + chosen_arr[:-1]] += 1

    return chosen

def _local_search(initial_set, deadline, max_iter=5000):
    """Local search using NumPy."""
    chosen_arr = np.zeros(1000, dtype=np.int32)
    n_chosen = 0
    in_set = np.zeros(1001, dtype=bool)
    counts = np.zeros(2001, dtype=np.int8)
    
    for x in initial_set:
        chosen_arr[n_chosen] = x
        n_chosen += 1
        in_set[x] = True
        
    # Init counts
    for i in range(n_chosen):
        a = chosen_arr[i]
        counts[a + chosen_arr[i:n_chosen]] += 1
        
    best_size = n_chosen
    best_set = list(chosen_arr[:n_chosen])
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try Add
        for x in range(1, 1001):
            if in_set[x]:
                continue
            if counts[2 * x] >= 2:
                continue
            if n_chosen > 0:
                sums = x + chosen_arr[:n_chosen]
                if np.any(counts[sums] >= 2):
                    continue
            
            # Add
            chosen_arr[n_chosen] = x
            n_chosen += 1
            in_set[x] = True
            counts[2 * x] += 1
            if n_chosen > 1:
                counts[x + chosen_arr[:n_chosen-1]] += 1
            
            best_size = n_chosen
            best_set = list(chosen_arr[:n_chosen])
            improved = True
            break
        
        if improved:
            continue
            
        # Try Swap
        if n_chosen > 0:
            idx = random.randint(0, n_chosen - 1)
            x = chosen_arr[idx]
            
            # Remove x
            chosen_arr[idx] = chosen_arr[n_chosen - 1]
            chosen_arr[n_chosen - 1] = 0
            n_chosen -= 1
            in_set[x] = False
            
            # Update counts
            counts[2 * x] -= 1
            if n_chosen > 0:
                counts[x + chosen_arr[:n_chosen]] -= 1
            
            # Try to add a random y
            swapped = False
            for _ in range(20):
                y = random.randint(1, 1000)
                if in_set[y]:
                    continue
                if counts[2 * y] >= 2:
                    continue
                if n_chosen > 0:
                    sums = y + chosen_arr[:n_chosen]
                    if np.any(counts[sums] >= 2):
                        continue
                
                # Add y
                chosen_arr[n_chosen] = y
                n_chosen += 1
                in_set[y] = True
                counts[2 * y] += 1
                if n_chosen > 1:
                    counts[y + chosen_arr[:n_chosen-1]] += 1
                    
                if n_chosen > best_size:
                    best_size = n_chosen
                    best_set = list(chosen_arr[:n_chosen])
                swapped = True
                break
            
            if not swapped:
                # Revert: Add x back
                chosen_arr[n_chosen] = x
                n_chosen += 1
                in_set[x] = True
                counts[2 * x] += 1
                if n_chosen > 1:
                    counts[x + chosen_arr[:n_chosen-1]] += 1

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    best_set = []
    best_size = 0
    
    # 1. Deterministic Greedy (Ascending)
    s1 = _try_greedy_fast(range(1, 1001), deadline)
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
        
    # 2. Deterministic Greedy (Descending)
    s2 = _try_greedy_fast(range(1000, 0, -1), deadline)
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
        
    # 3. Randomized Greedy + Local Search
    rng = random.Random(42)
    
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        order = list(range(1, 1001))
        rng.shuffle(order)
        s = _try_greedy_fast(order, deadline)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
        # Local search on the current best (or current s if it's better)
        if time.time() < deadline - 2.0:
            improved = _local_search(s, deadline - 2.0, max_iter=5000)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
                
    return best_set

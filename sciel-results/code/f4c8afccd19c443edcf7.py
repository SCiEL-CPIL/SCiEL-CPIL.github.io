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

def _local_search(initial_set, deadline, max_iter=10000):
    """Enhanced local search with 1-swaps and 2-swaps."""
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
    
    # Pre-compute list of all numbers 1..1000 for faster iteration
    all_numbers = np.arange(1, 1001, dtype=np.int32)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # 1. Try Add: find any valid addition
        # We scan all numbers, which is O(N) per attempt. 
        # To speed up, we can just pick random candidates first? 
        # Scanning is safer for finding *any* addition.
        for x in all_numbers:
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
            
        # 2. Try 1-Swap: remove one, add one
        if n_chosen > 0:
            # Try a few random removals
            for _ in range(5):
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
                for _ in range(100):
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
                else:
                    improved = True
                    break
        
        if improved:
            continue
            
        # 3. Try 2-Swap: remove two, add two
        # This is more expensive but can escape local minima
        if n_chosen >= 2:
            # Pick two random indices to remove
            idx1 = random.randint(0, n_chosen - 1)
            idx2 = random.randint(0, n_chosen - 1)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % n_chosen
            
            x1 = chosen_arr[idx1]
            x2 = chosen_arr[idx2]
            
            # Remove x1 and x2
            # Careful with indices if we remove one, the other shifts
            # Let's just mark them and rebuild? No, that's slow.
            # Let's do it carefully.
            
            # Remove x1
            chosen_arr[idx1] = chosen_arr[n_chosen - 1]
            n_chosen -= 1
            in_set[x1] = False
            counts[2 * x1] -= 1
            if n_chosen > 0:
                counts[x1 + chosen_arr[:n_chosen]] -= 1
            
            # Adjust idx2 if it was after idx1
            if idx2 > idx1:
                idx2 -= 1
            
            # Remove x2
            chosen_arr[idx2] = chosen_arr[n_chosen - 1]
            n_chosen -= 1
            in_set[x2] = False
            counts[2 * x2] -= 1
            if n_chosen > 0:
                counts[x2 + chosen_arr[:n_chosen]] -= 1
            
            # Try to add two new elements
            added = 0
            candidates = [random.randint(1, 1000) for _ in range(200)]
            random.shuffle(candidates)
            
            for y in candidates:
                if added == 2:
                    break
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
                added += 1
            
            if added == 2:
                if n_chosen > best_size:
                    best_size = n_chosen
                    best_set = list(chosen_arr[:n_chosen])
                improved = True
            else:
                # Revert: Add x1 and x2 back
                # Note: x1 and x2 are distinct
                for x in [x1, x2]:
                    chosen_arr[n_chosen] = x
                    n_chosen += 1
                    in_set[x] = True
                    counts[2 * x] += 1
                    if n_chosen > 1:
                        counts[x + chosen_arr[:n_chosen-1]] += 1

    return best_set

def _polish(set_arr, deadline):
    """Final polish: try to add more elements to an existing valid set."""
    chosen_arr = np.zeros(1000, dtype=np.int32)
    n_chosen = 0
    in_set = np.zeros(1001, dtype=bool)
    counts = np.zeros(2001, dtype=np.int8)
    
    for x in set_arr:
        chosen_arr[n_chosen] = x
        n_chosen += 1
        in_set[x] = True
        
    # Init counts
    for i in range(n_chosen):
        a = chosen_arr[i]
        counts[a + chosen_arr[i:n_chosen]] += 1
        
    all_numbers = np.arange(1, 1001, dtype=np.int32)
    
    # Try to add any valid element in a loop
    progress = True
    while progress:
        progress = False
        for x in all_numbers:
            if time.time() > deadline:
                return list(chosen_arr[:n_chosen])
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
            progress = True

    return list(chosen_arr[:n_chosen])

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
    
    while time.time() < deadline - 15.0:
        if time.time() > deadline - 15.0:
            break
            
        order = list(range(1, 1001))
        rng.shuffle(order)
        s = _try_greedy_fast(order, deadline - 15.0)
        
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
            
        # Local search on the current best (or current s if it's better)
        if time.time() < deadline - 18.0:
            improved = _local_search(s, deadline - 18.0, max_iter=20000)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
                
    # Final polish pass on the best set found
    if time.time() < deadline - 2.0:
        polished = _polish(best_set, deadline - 2.0)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(polished)
            
    return best_set

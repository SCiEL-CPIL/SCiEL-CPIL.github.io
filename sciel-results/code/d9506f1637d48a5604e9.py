import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 1000
    counts = np.zeros(2 * N + 1, dtype=np.int32)
    S = []
    in_set = np.zeros(N + 1, dtype=bool)
    
    def add_element(k):
        nonlocal counts, S, in_set
        # Check validity
        # S is a list, convert to array for speed?
        # If S is small, list is fine. But for NumPy check, we need array.
        # We can maintain S as a list and convert on the fly?
        # Or maintain S as a NumPy array?
        # Adding/removing from NumPy array is expensive (copy).
        # Better: Keep S as list. For check, use np.array(S).
        
        if counts[2 * k] > 1:
            return False
        
        if S:
            S_arr = np.array(S, dtype=np.int32)
            sums = k + S_arr
            # Check if any count > 1
            if np.any(counts[sums] > 1):
                return False
        
        # Commit
        in_set[k] = True
        S.append(k)
        counts[2 * k] += 1
        if S:
            # S includes k now? No, we appended k.
            # We need to update counts for k+i where i in OLD S.
            # The sums calculated above were for OLD S.
            # So we can just use those sums?
            # Yes, sums = k + OLD_S.
            # But we need to update counts for those sums.
            # We also need to update counts[2k].
            
            # Re-calculate sums for OLD S?
            # We have S_arr from OLD S.
            # So:
            counts[sums] += 1
            # Wait, S_arr is OLD S.
            # sums = k + OLD_S.
            # This is correct.
            
            # But what if S was empty?
            # Then S_arr is empty, sums is empty.
            # counts[sums] += 1 does nothing.
            # counts[2*k] += 1 is done.
            
        return True

    def remove_element(k):
        nonlocal counts, S, in_set
        in_set[k] = False
        S.remove(k)
        counts[2 * k] -= 1
        if S:
            S_arr = np.array(S, dtype=np.int32)
            sums = k + S_arr
            counts[sums] -= 1
        return True

    # 1. Initial Greedy
    for k in range(1, N + 1):
        if in_set[k]:
            continue
        if add_element(k):
            pass

    best_size = len(S)
    best_S = S.copy()
    best_counts = counts.copy()
    best_in_set = in_set.copy()

    # 2. Local Search
    # Use NumPy for candidate checking
    # Pre-allocate arrays?
    
    while time.time() < deadline:
        # Try to add
        added = False
        # Sample random candidates
        # Instead of 50, maybe 10?
        for _ in range(10):
            k = random.randint(1, N)
            if in_set[k]:
                continue
            
            # Check
            if counts[2 * k] > 1:
                continue
            if S:
                S_arr = np.array(S, dtype=np.int32)
                sums = k + S_arr
                if np.any(counts[sums] > 1):
                    continue
            
            # Valid
            add_element(k)
            added = True
            if len(S) > best_size:
                best_size = len(S)
                best_S = S.copy()
                best_counts = counts.copy()
                best_in_set = in_set.copy()
            break
        
        if not added:
            # Try swap
            swapped = False
            if S:
                for _ in range(5):
                    i = random.choice(S)
                    j = random.randint(1, N)
                    if in_set[j]:
                        continue
                    
                    remove_element(i)
                    
                    # Check j
                    if counts[2 * j] > 1:
                        add_element(i)
                        continue
                    if S:
                        S_arr = np.array(S, dtype=np.int32)
                        sums = j + S_arr
                        if np.any(counts[sums] > 1):
                            add_element(i)
                            continue
                    
                    add_element(j)
                    swapped = True
                    break
            
            if not swapped:
                if S:
                    i = random.choice(S)
                    remove_element(i)

    return best_S

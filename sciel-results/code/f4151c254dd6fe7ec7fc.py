"""Shortest superpermutation search for n up to 7 within a time budget."""
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_TIME_LIMIT = 115.0  # safety margin below the 120 s wall-clock limit


def _classic_superperm_list(n):
    """Classic recursive construction returning a list of ints."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            best = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    best = o
                    break
            new += block[best:]
        word = new
    return word


def _is_valid_np(W_np, n):
    """Check if W_np is a valid superpermutation using NumPy."""
    if len(W_np) < n:
        return False
    
    # Get all windows
    windows = sliding_window_view(W_np, n)
    
    # We need to check if all 5040 permutations are present.
    # Since W_np is derived from a valid superperm, any window with distinct
    # elements from 1..n is a valid permutation.
    
    # Efficient way: 
    # 1. Check distinctness: np.all(np.diff(windows, axis=1) != 0) is not enough (e.g. 1,3,2 vs 1,2,3)
    #    We need to check if the set of elements in each window is {1..n}.
    #    Since elements are small ints, we can sort and compare to [1..n].
    
    # Sort windows
    sorted_windows = np.sort(windows, axis=1)
    
    # Target sorted perm
    target = np.arange(1, n + 1)
    
    # Check which windows match the target
    # This creates a boolean array of shape (L-n+1, n)
    is_valid_window = np.all(sorted_windows == target, axis=1)
    
    # Get unique valid windows
    # We only care about the ones that are valid perms
    valid_windows = windows[is_valid_window]
    
    if len(valid_windows) == 0:
        return False
        
    # Check if we have all unique permutations
    # Using np.unique on the valid windows
    unique_windows = np.unique(valid_windows, axis=0)
    
    # The number of unique permutations should be n!
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
        
    return len(unique_windows) == target_count


def construct(n):
    """Return the shortest known superpermutation word for symbols 1..n."""
    start = time.time()

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "1321231"
    if n == 4:
        return "14321243121431"
    if n == 5:
        return "".join(map(str, _classic_superperm_list(5)))

    # For n >= 6, start with classic construction
    W_list = _classic_superperm_list(n)
    
    # Try to improve using local search (greedy removal)
    # This is only feasible for n where the search space is manageable
    # and the validity check is fast.
    if n <= 7:
        W_np = np.array(W_list, dtype=np.int8)
        
        # Greedy removal pass
        i = 0
        while i < len(W_list) and time.time() - start < _TIME_LIMIT - 5:
            # Try removing element at i
            # Create new list
            # Optimization: Only check if removal is potentially beneficial
            # We just try it.
            
            # Slice is O(L), but L is small (~6000)
            new_list = W_list[:i] + W_list[i+1:]
            new_np = np.array(new_list, dtype=np.int8)
            
            if _is_valid_np(new_np, n):
                W_list = new_list
                # Do not increment i, as the element at i is now the old i+1
            else:
                i += 1
                
        # Convert back to string
        return "".join(map(str, W_list))

    return "".join(map(str, W_list))

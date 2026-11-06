"""Rewritten program for 4-term-AP-free subset of [1,512]."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 512

def _has_4ap_numpy(arr, x_idx):
    """Check if adding element at x_idx (0-indexed) to the boolean array creates a 4-term AP."""
    # Case 1: x is the 4th term: x-3d, x-2d, x-d in set
    for d in range(1, x_idx // 3 + 1):
        if arr[x_idx - 3*d] and arr[x_idx - 2*d] and arr[x_idx - d]:
            return True
    # Case 2: x is the 3rd term: x-2d, x-d, x+d in set
    for d in range(1, x_idx // 2 + 1):
        if arr[x_idx - 2*d] and arr[x_idx - d] and arr[x_idx + d]:
            return True
    # Case 3: x is the 2nd term: x-d, x+d, x+2d in set
    for d in range(1, x_idx + 1):
        if x_idx + 2*d >= N:
            break
        if arr[x_idx - d] and arr[x_idx + d] and arr[x_idx + 2*d]:
            return True
    # Case 4: x is the 1st term: x+d, x+2d, x+3d in set
    for d in range(1, (N - 1 - x_idx) // 3 + 1):
        if arr[x_idx + d] and arr[x_idx + 2*d] and arr[x_idx + 3*d]:
            return True
    return False

def _greedy_numpy(order):
    """Greedy construction using numpy boolean array."""
    arr = np.zeros(N, dtype=bool)
    count = 0
    for x in order:
        x_idx = x - 1
        if not _has_4ap_numpy(arr, x_idx):
            arr[x_idx] = True
            count += 1
    return arr, count

def _local_search_numpy(arr, max_iter=100, time_limit=None):
    """Improve the set using swap moves: remove one element, add one or two others."""
    current_size = arr.sum()
    start = time.time()
    
    for it in range(max_iter):
        if time_limit and time.time() > time_limit:
            break
        improved = False
        
        # Try removing one, adding two (net +1)
        for rem in range(N):
            if not arr[rem]:
                continue
            arr[rem] = False
            trial_size = current_size - 1
            
            # Try adding pairs
            for add1 in range(N):
                if add1 == rem or arr[add1]:
                    continue
                if _has_4ap_numpy(arr, add1):
                    continue
                arr[add1] = True
                
                for add2 in range(add1 + 1, N):
                    if add2 == rem or arr[add2]:
                        continue
                    if _has_4ap_numpy(arr, add2):
                        continue
                    arr[add2] = True
                    new_size = trial_size + 2
                    if new_size > current_size:
                        current_size = new_size
                        improved = True
                        break
                    else:
                        arr[add2] = False
                if improved:
                    break
                else:
                    arr[add1] = False
            if improved:
                break
            else:
                arr[rem] = True
        
        if not improved:
            # Try random kicks: remove 1-2 random elements, add 1-2 random elements
            kick_size = random.randint(1, 2)
            to_remove = random.sample([i for i in range(N) if arr[i]], min(kick_size, current_size))
            for i in to_remove:
                arr[i] = False
            current_size -= len(to_remove)
            
            # Try to add back
            candidates = [i for i in range(N) if not arr[i]]
            random.shuffle(candidates)
            for add in candidates[:kick_size + 1]:
                if not _has_4ap_numpy(arr, add):
                    arr[add] = True
                    current_size += 1
        
        if time_limit and time.time() > time_limit:
            break
    
    return arr, current_size

def _construct_base():
    """Base construction using a known good pattern."""
    # Use a simple construction: numbers with only 0s and 1s in base 4
    # This gives a 4-AP-free set
    arr = np.zeros(N, dtype=bool)
    for x in range(1, N + 1):
        # Check if x in base 4 uses only digits 0 and 1
        y = x
        valid = True
        while y > 0:
            if y % 4 > 1:
                valid = False
                break
            y //= 4
        if valid:
            arr[x - 1] = True
    return arr, arr.sum()

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Start with base construction
    best_arr, best_size = _construct_base()
    
    # Try greedy with different orderings
    orderings = [
        list(range(1, N + 1)),  # natural
        list(range(N, 0, -1)),  # reverse
    ]
    
    # Add some random orderings
    for _ in range(30):
        order = list(range(1, N + 1))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline - 5.0:
            break
        arr, size = _greedy_numpy(order)
        if size > best_size:
            best_arr = arr.copy()
            best_size = size
    
    # Local search on the best set
    if time.time() < deadline - 2.0:
        arr, size = _local_search_numpy(best_arr.copy(), max_iter=50, time_limit=deadline - 2.0)
        if size > best_size:
            best_arr = arr
            best_size = size
    
    # Multiple rounds of local search with random kicks
    while time.time() < deadline - 1.0:
        arr, size = _local_search_numpy(best_arr.copy(), max_iter=30, time_limit=deadline - 1.0)
        if size > best_size:
            best_arr = arr
            best_size = size
        else:
            break
    
    # Convert to list
    result = [i + 1 for i in range(N) if best_arr[i]]
    return sorted(result)
# EVOLVE-BLOCK-END

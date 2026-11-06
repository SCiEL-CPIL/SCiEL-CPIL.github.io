import time
import numpy as np

def _subset_sums_distinct_fast(elements):
    """
    Check if all subset sums are distinct using a set-based approach.
    Returns True if all 2^n subset sums are unique.
    """
    sums = {0}
    for e in elements:
        new_sums = set()
        for s in sums:
            ns = s + e
            if ns in sums:
                return False
            new_sums.add(ns)
        if len(new_sums) != len(sums):
            return False
        sums.update(new_sums)
    return True

def _subset_sums_distinct_numpy(elements):
    """
    Check if all subset sums are distinct using numpy for speed.
    Returns True if all 2^n subset sums are unique.
    """
    n = len(elements)
    if n == 0:
        return True
    # Generate all subset sums using numpy
    # Use bit manipulation to generate subset sums efficiently
    # For n=12, there are 4096 subsets
    # We can compute all subset sums at once
    
    # Create a matrix where each row represents a subset
    # Use the trick: subset sum = dot product of element vector with subset indicator vector
    # But for n=12, even the simple approach is fast enough
    
    # Actually, let's use a smarter approach:
    # Generate all subset sums incrementally
    sums = np.array([0], dtype=np.int64)
    for e in elements:
        new_sums = sums + e
        sums = np.concatenate([sums, new_sums])
        # Check for duplicates
        if len(np.unique(sums)) != len(sums):
            return False
    return True

def _find_minimal_superincreasing(n):
    """
    Find the minimal superincreasing sequence of length n.
    A superincreasing sequence guarantees distinct subset sums.
    The minimal such sequence starts with 1, and each next element
    is 1 + sum of all previous elements.
    This gives: 1, 2, 4, 8, ..., 2^(n-1)
    """
    arr = [1]
    for i in range(1, n):
        arr.append(1 + sum(arr))
    return arr

def construct():
    """
    Returns a list of 12 distinct positive integers with distinct subset sums,
    minimizing the largest element.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 12
    
    # Phase 1: Start with powers of 2 (minimal superincreasing sequence)
    arr = [1 << i for i in range(n)]
    best_arr = list(arr)
    best_max = max(arr)
    
    # Verify initial solution
    if not _subset_sums_distinct_fast(best_arr):
        raise ValueError("Initial solution invalid")
    
    # Phase 2: Greedy construction from scratch - try to find a better sequence
    # At each step, pick the smallest positive integer that maintains distinct subset sums
    if time.time() < deadline:
        greedy_arr = []
        used = set()
        for i in range(n):
            if time.time() > deadline:
                break
            candidate = 1
            found = False
            while not found:
                if time.time() > deadline:
                    break
                if candidate not in used:
                    test = greedy_arr + [candidate]
                    if _subset_sums_distinct_fast(test):
                        greedy_arr.append(candidate)
                        used.add(candidate)
                        found = True
                candidate += 1
            if time.time() > deadline:
                break
        if len(greedy_arr) == n and max(greedy_arr) < best_max:
            best_arr = greedy_arr
            best_max = max(greedy_arr)
    
    # Phase 3: Local search - try to reduce elements
    # Use the numpy-based check for speed
    arr = list(best_arr)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1, -1, -1):  # From largest index to smallest
            if time.time() > deadline:
                break
            old_val = arr[i]
            if old_val <= 1:
                continue
            
            # Try reducing by various amounts
            for delta in range(1, min(old_val, 20)):
                if time.time() > deadline:
                    break
                new_val = old_val - delta
                if new_val <= 0:
                    break
                if any(arr[j] == new_val for j in range(n) if j != i):
                    continue
                arr[i] = new_val
                if _subset_sums_distinct_numpy(arr):
                    improved = True
                    if max(arr) < best_max:
                        best_max = max(arr)
                        best_arr = list(arr)
                    break
                else:
                    arr[i] = old_val
                    break
            
            # Try larger jumps
            if time.time() < deadline:
                for delta in [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
                    if time.time() > deadline:
                        break
                    new_val = old_val - delta
                    if new_val <= 0:
                        continue
                    if any(arr[j] == new_val for j in range(n) if j != i):
                        continue
                    arr[i] = new_val
                    if _subset_sums_distinct_numpy(arr):
                        improved = True
                        if max(arr) < best_max:
                            best_max = max(arr)
                            best_arr = list(arr)
                        break
                    else:
                        arr[i] = old_val
    
    # Phase 4: Final polish - try to reduce the maximum element specifically
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            max_idx = best_arr.index(max(best_arr))
            old_max = best_arr[max_idx]
            for new_val in range(old_max - 1, max(0, old_max - 500), -1):
                if time.time() > deadline:
                    break
                if any(best_arr[j] == new_val for j in range(n) if j != max_idx):
                    continue
                test_arr = list(best_arr)
                test_arr[max_idx] = new_val
                if _subset_sums_distinct_numpy(test_arr):
                    best_arr = test_arr
                    best_max = new_val
                    break
    
    # Final validation
    if _subset_sums_distinct_fast(best_arr):
        return best_arr
    else:
        return [1 << i for i in range(n)]

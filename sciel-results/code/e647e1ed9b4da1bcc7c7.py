import time
import numpy as np

def construct():
    """
    Returns a list of 12 distinct positive integers such that all 4096 subset sums are unique.
    The integers are chosen to minimize the largest element.
    
    Strategy:
    1. Start with the greedy construction (powers of 2).
    2. Apply iterative local search to reduce elements while maintaining distinct subset sums.
    3. Use efficient numpy-based subset sum computation for speed.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def _all_subset_sums_distinct(arr):
        """Check that all 2^n subset sums are distinct using numpy for speed."""
        n = len(arr)
        # Generate all subset sums efficiently
        # Use the fact that subset sums can be built iteratively
        sums = np.array([0])
        for i in range(n):
            new_sums = sums + arr[i]
            sums = np.concatenate([sums, new_sums])
        # Check for duplicates
        if len(sums) != len(np.unique(sums)):
            return False
        return True

    def _greedy_construct(n):
        """
        Greedy construction: iteratively select the smallest positive integer
        not yet used that preserves the distinctness of all current subset sums.
        """
        arr = []
        used = set()
        for _ in range(n):
            candidate = 1
            while True:
                if candidate in used:
                    candidate += 1
                    continue
                # Test if adding candidate preserves distinctness
                test_arr = arr + [candidate]
                if _all_subset_sums_distinct(test_arr):
                    arr.append(candidate)
                    used.add(candidate)
                    break
                candidate += 1
        return arr

    # Start with greedy construction (should yield powers of 2)
    arr = _greedy_construct(12)
    best_max = max(arr)
    best_arr = list(arr)

    # Verify initial solution is valid
    assert _all_subset_sums_distinct(best_arr)

    # Sort in descending order to prioritize reducing larger elements
    arr_sorted = sorted(arr, reverse=True)
    arr = arr_sorted

    # Local search: iteratively try to reduce elements
    improved = True
    max_iterations = 1000
    iteration = 0

    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try reducing each element from largest to smallest
        for i in range(11, -1, -1):
            if time.time() > deadline:
                break
            old_val = arr[i]
            if old_val <= 1:
                continue
            
            # Try reducing by various amounts, starting from small
            for delta in range(1, min(old_val, 500)):
                if time.time() > deadline:
                    break
                new_val = old_val - delta
                if new_val <= 0:
                    break
                
                # Check new_val is not equal to any other element
                if any(arr[j] == new_val for j in range(12) if j != i):
                    continue
                
                arr[i] = new_val
                if _all_subset_sums_distinct(arr):
                    improved = True
                    if max(arr) < best_max:
                        best_max = max(arr)
                        best_arr = list(arr)
                        # Keep the current improved state
                    else:
                        # Even if max didn't improve, this move might enable future improvements
                        # Keep it if it doesn't make things worse
                        if max(arr) <= best_max:
                            pass  # Keep the change
                        else:
                            arr[i] = old_val
                        break
                else:
                    arr[i] = old_val
                    break  # If reducing by delta fails, try next element

    # Additional refinement: try larger reductions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(11, -1, -1):
            if time.time() > deadline:
                break
            old_val = arr[i]
            if old_val <= 1:
                continue
            # Try reducing by larger amounts
            for delta in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
                if time.time() > deadline:
                    break
                new_val = old_val - delta
                if new_val <= 0:
                    break
                if any(arr[j] == new_val for j in range(12) if j != i):
                    continue
                arr[i] = new_val
                if _all_subset_sums_distinct(arr):
                    improved = True
                    if max(arr) < best_max:
                        best_max = max(arr)
                        best_arr = list(arr)
                else:
                    arr[i] = old_val
                    break  # If reducing by delta fails, smaller deltas might work but let's try next element

    # Final check
    if _all_subset_sums_distinct(best_arr):
        return best_arr
    # Fallback to powers of 2
    return [1 << i for i in range(12)]

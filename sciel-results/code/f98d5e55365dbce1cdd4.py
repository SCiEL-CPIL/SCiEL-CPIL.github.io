import time

def construct():
    """
    Returns a list of 12 distinct positive integers such that all 4096 subset sums are unique.
    The integers are chosen to minimize the largest element.
    
    Strategy:
    1. Start with the greedy construction: iteratively select the smallest positive integer
       that preserves the distinctness of all current subset sums. This is equivalent to
       powers of 2 for this problem structure.
    2. Apply local search to try to reduce elements while maintaining distinct subset sums.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    def _all_subset_sums_distinct(arr):
        """Check that all 2^n subset sums are distinct."""
        n = len(arr)
        sums = set()
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += arr[i]
            if s in sums:
                return False
            sums.add(s)
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

    # Local search: try to reduce individual elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try reducing from largest to smallest
        for i in range(11, -1, -1):
            if time.time() > deadline:
                break
            old_val = arr[i]
            if old_val <= 1:
                continue
            # Try reducing by 1
            new_val = old_val - 1
            if new_val <= 0:
                continue
            # Check new_val is not equal to any other element
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

    # Additional refinement: try reducing by larger amounts
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(11, -1, -1):
            if time.time() > deadline:
                break
            old_val = arr[i]
            if old_val <= 1:
                continue
            # Try reducing by various amounts
            for delta in [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040]:
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

import time
import random

def _is_valid(arr):
    """Check if arr satisfies the cardinality-separated subset sum property."""
    n = len(arr)
    if n == 0:
        return True
    # For each k, compute min and max sum of k-element subsets
    # Condition: max_k_sum < min_{k+1}_sum for all k
    sorted_arr = sorted(arr)
    # Prefix sums
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + sorted_arr[i]
    
    for k in range(n):
        # Min sum of (k+1)-subset: sum of first (k+1) elements
        min_next = prefix[k + 1]
        # Max sum of k-subset: sum of last k elements
        if k == 0:
            max_curr = 0
        else:
            max_curr = prefix[n] - prefix[n - k]
        if max_curr >= min_next:
            return False
    return True

def _search_consecutive():
    """Find largest n with consecutive integers that work."""
    for n in range(60, 0, -1):
        # For consecutive a, a+1, ..., a+n-1:
        # Max k-sum = sum of last k = k*(a+n-1) - k*(k-1)//2
        # Min (k+1)-sum = sum of first (k+1) = (k+1)*a + k*(k+1)//2
        # Need: k*(a+n-1) - k*(k-1)//2 < (k+1)*a + k*(k+1)//2
        # k*a + k*(n-1) - k*(k-1)//2 < (k+1)*a + k*(k+1)//2
        # k*(n-1) - k*(k-1)//2 < a + k*(k+1)//2
        # a > k*(n-1) - k*(k-1)//2 - k*(k+1)//2 = k*(n-1) - k^2
        min_a_needed = 1
        for k in range(1, n):
            req = k * (n - 1) - k * k
            if req + 1 > min_a_needed:
                min_a_needed = req + 1
        max_a = 256 - n + 1
        if min_a_needed <= max_a:
            return list(range(min_a_needed, min_a_needed + n))
    return None

def _try_nonconsecutive(n, deadline=None):
    """Try to find a valid set of size n using non-consecutive integers."""
    if deadline is not None and time.monotonic() > deadline:
        return None
    
    # Start with consecutive and try to optimize
    # The constraints are: for each k, max_k_sum < min_{k+1}_sum
    # This means we need enough "gap" in the middle
    
    # Parametric approach: try different distributions
    # The tightest constraint is around k = n//2
    # Let's try to spread elements more in the middle
    
    best = None
    rng = random.Random(12345)
    
    # Try several random perturbations of consecutive
    base_a = 1
    for n_try in range(n, min(n, 40), -1):
        if deadline is not None and time.monotonic() > deadline:
            break
        
        # Find minimum a for consecutive
        min_a = 1
        for k in range(1, n_try):
            req = k * (n_try - 1) - k * k
            if req + 1 > min_a:
                min_a = req + 1
        max_a = 256 - n_try + 1
        
        if min_a > max_a:
            continue
        
        # Try consecutive
        cons = list(range(min_a, min_a + n_try))
        if _is_valid(cons):
            if best is None or n_try > len(best):
                best = cons
            break
        
        # Try non-consecutive: add slack to middle elements
        for attempt in range(200):
            if deadline is not None and time.monotonic() > deadline:
                break
            # Start from consecutive and perturb
            arr = list(range(min_a, min_a + n_try))
            # Add random gaps to middle elements
            for _ in range(10):
                i = rng.randint(1, n_try - 2)
                arr[i] += rng.randint(1, 5)
            arr.sort()
            # Check bounds
            if arr[0] < 1 or arr[-1] > 256 or len(set(arr)) != n_try:
                continue
            if _is_valid(arr):
                if best is None or n_try > len(best):
                    best = arr
                break
    
    return best

def _local_search(arr, deadline):
    """Try to improve a valid set by swapping elements."""
    arr = sorted(arr)
    n = len(arr)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        arr_set = set(arr)
        for i in range(n):
            for v in range(1, 257):
                if v in arr_set:
                    continue
                new_arr = arr[:i] + [v] + arr[i+1:]
                new_arr.sort()
                if _is_valid(new_arr):
                    # Try to grow
                    grown = _try_grow(new_arr, deadline)
                    if len(grown) > len(arr):
                        arr = grown
                        n = len(arr)
                        improved = True
                        break
            if improved:
                break
    return arr

def _try_grow(arr, deadline):
    """Try to add elements to arr."""
    arr = sorted(arr)
    n = len(arr)
    arr_set = set(arr)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        for v in range(1, 257):
            if v in arr_set:
                continue
            new_arr = arr + [v]
            new_arr.sort()
            if _is_valid(new_arr):
                arr = new_arr
                arr_set.add(v)
                n += 1
                improved = True
                break
    return arr

def construct():
    t0 = time.monotonic()
    deadline = t0 + 80.0  # Safety margin before 90s limit
    best = None
    
    def update_best(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            best = list(arr)
    
    # 1. Consecutive search (fast, guaranteed)
    res = _search_consecutive()
    if res:
        update_best(res)
    
    # 2. Try to beat it with non-consecutive
    if best is not None:
        target_n = len(best) + 1
        # Try sizes from target_n down
        for n_try in range(target_n, min(target_n, 45), -1):
            if time.monotonic() > deadline - 5:
                break
            res = _try_nonconsecutive(n_try, deadline)
            if res:
                update_best(res)
                break
    
    # 3. Local search to try to grow
    if best is not None and time.monotonic() < deadline - 5:
        polished = _local_search(best, deadline)
        update_best(polished)
    
    # 4. Random restarts
    rng = random.Random(42)
    if best is not None and time.monotonic() < deadline - 3:
        for _ in range(50):
            if time.monotonic() > deadline - 2:
                break
            k = rng.randint(5, 15)
            start = rng.sample(range(1, 257), k)
            start.sort()
            if _is_valid(start):
                grown = _try_grow(start, deadline)
                update_best(grown)
    
    # Final validity check
    if best is not None:
        if not _is_valid(best):
            best = _search_consecutive()
            if best is None:
                best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    if best is None:
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Ensure all are Python ints
    return [int(x) for x in best]

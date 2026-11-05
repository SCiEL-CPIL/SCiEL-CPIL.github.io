import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe (90s total budget)
    
    n = 12
    # Initial best: powers of 2
    best_set = [1 << i for i in range(n)]
    best_max = max(best_set)  # 2048
    
    def _is_valid(arr):
        """Check if all 2^n subset sums are distinct. Fast set-based check."""
        if len(arr) != n:
            return False
        if len(set(arr)) != n:
            return False
        if any(x <= 0 for x in arr):
            return False
        if any(x > 1048576 for x in arr):
            return False
        sums = {0}
        for x in arr:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums = sums | new_sums
        return True

    def _find_smallest_valid(arr, idx, max_val):
        """Find the smallest value for arr[idx] that keeps validity. Returns -1 if none."""
        others = [arr[j] for j in range(n) if j != idx]
        # Precompute subset sums of others
        others_sums = {0}
        for x in others:
            new_sums = set()
            for s in others_sums:
                ns = s + x
                if ns in others_sums or ns in new_sums:
                    return -1  # others alone are invalid (shouldn't happen)
                new_sums.add(ns)
            others_sums = others_sums | new_sums
        
        # For each candidate v, check that {s + v for s in others_sums} is disjoint from others_sums
        # and v is not in others
        for v in range(1, max_val + 1):
            if v in others:
                continue
            # Check disjointness: for all s in others_sums, s+v not in others_sums
            valid = True
            for s in others_sums:
                if (s + v) in others_sums:
                    valid = False
                    break
            if valid:
                return v
        return -1

    def _local_search(arr, max_time):
        """Try to reduce elements via local search using smallest-valid-value approach."""
        arr = arr[:]
        improved = True
        while improved and time.time() < max_time:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > max_time:
                    break
                old_val = arr[idx]
                # Find smallest valid value; if it's < old_val, replace
                new_val = _find_smallest_valid(arr, idx, old_val - 1)
                if new_val != -1 and new_val < old_val:
                    arr[idx] = new_val
                    improved = True
        return arr

    def _greedy_construct():
        """
        Greedy construction: iteratively select the smallest positive integer
        not yet used which preserves distinctness of all subset sums.
        This is the core strategy from the directive.
        """
        current = []
        used = set()
        sums = {0}
        
        for i in range(n):
            # Find smallest x > 0 not in used such that all subset sums remain distinct
            x = 1
            while True:
                if x in used:
                    x += 1
                    continue
                
                # Check if adding x preserves distinctness:
                # For all s in sums, s+x must not be in sums
                valid = True
                for s in sums:
                    if (s + x) in sums:
                        valid = False
                        break
                if valid:
                    current.append(x)
                    used.add(x)
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + x)
                    sums = new_sums
                    break
                x += 1
                if x > 100000:
                    break
            
            if x > 100000:
                break
        
        return current if len(current) == n else None

    # Phase 1: Try greedy construction
    if time.time() < deadline - 30:
        greedy_result = _greedy_construct()
        if greedy_result is not None:
            # Polish it with local search
            polished = _local_search(greedy_result, deadline - 10)
            if _is_valid(polished):
                if max(polished) < best_max:
                    best_max = max(polished)
                    best_set = polished

    # Phase 2: Beam search with local search polish
    if time.time() < deadline - 20:
        # Beam search
        # State: (arr, sums_set)
        # We'll maintain a beam of partial sequences
        beam = [([1], {0, 1})]
        
        for step in range(1, n):
            if time.time() > deadline - 10:
                break
            
            next_beam = []
            for arr, sums in beam:
                # Find candidates that work
                candidates = []
                for x in range(1, 5000):
                    if x in arr:
                        continue
                    # Check collision: for all s in sums, s+x not in sums
                    valid = True
                    for s in sums:
                        if (s + x) in sums:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        if len(candidates) >= 30:
                            break
                
                for x in candidates:
                    new_sums = set(sums)
                    for s in sums:
                        new_sums.add(s + x)
                    next_beam.append((arr + [x], new_sums))
            
            # Sort by max element and keep top beam_width
            next_beam.sort(key=lambda item: max(item[0]))
            beam_width = 8
            beam = next_beam[:beam_width]
        
        # Polish final beam entries
        for arr, sums in beam:
            if time.time() > deadline - 5:
                break
            if len(arr) == n:
                polished = _local_search(arr, deadline - 2)
                if _is_valid(polished):
                    if max(polished) < best_max:
                        best_max = max(polished)
                        best_set = polished

    # Phase 3: More local search passes on best_set
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            polished = _local_search(best_set, deadline - 3)
            if _is_valid(polished):
                if max(polished) < best_max:
                    best_max = max(polished)
                    best_set = polished

    # Phase 4: Try greedy from different starting points
    if time.time() < deadline - 10:
        # Try greedy starting with different first elements
        for first_val in [1, 2, 3, 4, 5, 6, 7, 8]:
            if time.time() > deadline - 5:
                break
            current = [first_val]
            used = {first_val}
            sums = {0, first_val}
            success = True
            for i in range(1, n):
                x = 1
                while True:
                    if x in used:
                        x += 1
                        continue
                    valid = True
                    for s in sums:
                        if (s + x) in sums:
                            valid = False
                            break
                    if valid:
                        current.append(x)
                        used.add(x)
                        new_sums = set(sums)
                        for s in sums:
                            new_sums.add(s + x)
                        sums = new_sums
                        break
                    x += 1
                    if x > 200000:
                        success = False
                        break
                if not success:
                    break
            
            if success and len(current) == n:
                polished = _local_search(current, deadline - 3)
                if _is_valid(polished):
                    if max(polished) < best_max:
                        best_max = max(polished)
                        best_set = polished

    return best_set

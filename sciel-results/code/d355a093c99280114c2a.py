import time
import random

def _is_valid(arr):
    n = len(arr)
    if n < 1 or n > 60:
        return False
    s = set()
    for v in arr:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v < 1 or v > 256:
            return False
        if v in s:
            return False
        s.add(v)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    seen = 0
    for k in range(n + 1):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _check_add(arr, candidate):
    """Check if adding candidate to arr preserves validity. Returns True if valid."""
    if candidate in arr:
        return False
    n = len(arr)
    new_n = n + 1
    dp = [0] * (new_n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(new_n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    for k in range(new_n, 0, -1):
        dp[k] |= (dp[k - 1] << candidate)
    seen = 0
    for k in range(new_n + 1):
        if dp[k] & seen:
            return False
        seen |= dp[k]
    return True

def _grow_greedy(start, deadline, rng=None):
    """Greedy growth with collision-aware ordering."""
    if rng is None:
        rng = random.Random(42)
    cur = list(start)
    cur_set = set(cur)
    candidates = [c for c in range(1, 257) if c not in cur_set]
    
    while time.monotonic() < deadline:
        # Score each candidate by how "safe" it is
        # Try candidates in a mix of small-first and random order
        # First pass: try small values (they're generally safer)
        added = False
        for c in sorted(candidates):
            if c not in cur_set:
                if _check_add(cur, c):
                    cur.append(c)
                    cur_set.add(c)
                    candidates.remove(c)
                    added = True
                    break
        if added:
            continue
        # Second pass: try random order
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        for c in shuffled:
            if c not in cur_set:
                if _check_add(cur, c):
                    cur.append(c)
                    cur_set.add(c)
                    candidates.remove(c)
                    added = True
                    break
        if not added:
            break
    return cur

def _local_search(cur, deadline, rng):
    """Try to improve by swapping elements out for better ones."""
    cur = list(cur)
    cur_set = set(cur)
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        # Try removing one element and adding two (net +1)
        for i in range(len(cur)):
            if time.monotonic() > deadline:
                break
            removed = cur[i]
            remaining = [x for x in cur if x != removed]
            rem_set = set(remaining)
            # Try to add two new elements
            candidates = [c for c in range(1, 257) if c not in rem_set and c != removed]
            # Try pairs
            added_pair = False
            for c1 in candidates:
                if time.monotonic() > deadline:
                    break
                if _check_add(remaining, c1):
                    new_set1 = rem_set | {c1}
                    for c2 in candidates:
                        if c2 == c1 or c2 in new_set1:
                            continue
                        if _check_add(remaining + [c1], c2):
                            new_cur = remaining + [c1, c2]
                            # Verify full validity
                            if _is_valid(new_cur):
                                cur = new_cur
                                cur_set = set(cur)
                                improved = True
                                added_pair = True
                                break
                        if added_pair:
                            break
                if added_pair:
                    break
            if improved:
                break
    return cur

def construct():
    start_time = time.monotonic()
    deadline = start_time + 85.0  # safety margin
    
    best = []
    
    # Strategy 1: Start with superincreasing base and grow
    if time.monotonic() < deadline - 10:
        powers = [1, 2, 4, 8, 16, 32, 64, 128]
        if _is_valid(powers):
            result = _grow_greedy(powers, deadline, random.Random(1))
            if len(result) > len(best):
                best = result
    
    # Strategy 2: Start small and grow
    for start_val in [1, 2, 3, 5, 8, 10, 13, 20]:
        if time.monotonic() > deadline - 5:
            break
        cur = [start_val]
        result = _grow_greedy(cur, deadline, random.Random(start_val))
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Start with moderately spaced values
    if time.monotonic() < deadline - 10:
        for start_set in [
            [10, 20, 30, 40, 50, 60, 70, 80],
            [5, 15, 25, 35, 45, 55, 65, 75],
            [1, 3, 7, 15, 31, 63, 127],
            [2, 5, 12, 25, 50, 100],
            [1, 4, 9, 16, 25, 36, 49],
        ]:
            if not _is_valid(start_set):
                continue
            result = _grow_greedy(start_set, deadline, random.Random(99))
            if len(result) > len(best):
                best = result
    
    # Strategy 4: Local search on best found
    if best and time.monotonic() < deadline - 5:
        improved = _local_search(best, deadline, random.Random(123))
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if best:
        if not _is_valid(best):
            # Fallback to a known-valid small set
            best = list(range(128, 136))
    
    # Ensure we return a list of Python ints
    return list(best)

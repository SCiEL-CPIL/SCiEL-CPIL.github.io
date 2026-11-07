import time
import random

def _is_valid(arr):
    """Check: distinct ints in 1..256, size 1..60, cardinality-separated subset sums."""
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
    # dp[k] is a bitset: bit j set means sum j is achievable with exactly k elements
    max_sum = sum(arr)
    if max_sum > 256 * 60:  # safety
        return False
    dp = [0] * (n + 1)
    dp[0] = 1  # empty subset, sum 0
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    # Check no sum appears in two different cardinalities
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
        added = False
        # First pass: try small values (they're generally safer)
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
    best = list(cur)
    best_size = len(cur)
    cur_set = set(cur)
    
    # Try 1-for-1 swaps
    while time.monotonic() < deadline - 2:
        improved = False
        for i in range(len(cur)):
            if time.monotonic() > deadline - 2:
                break
            for c in range(1, 257):
                if c in cur_set:
                    continue
                # Try replacing cur[i] with c
                new_cur = cur[:i] + [c] + cur[i+1:]
                if _is_valid(new_cur):
                    # Check if we can grow further from here
                    grown = _grow_greedy(new_cur, min(deadline - 2, time.monotonic() + 5), rng)
                    if len(grown) > best_size:
                        best = grown
                        best_size = len(grown)
                        cur = list(grown)
                        cur_set = set(grown)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best

def construct():
    start_time = time.monotonic()
    deadline = start_time + 85.0  # safety margin for 90s limit
    
    rng = random.Random(42)
    
    # Try different starting strategies
    start_strategies = [
        [1],
        [1, 2],
        [1, 2, 4],
        [1, 2, 4, 8],
        [1, 2, 4, 8, 16],
        [1, 2, 4, 8, 16, 32],
        [1, 2, 4, 8, 16, 32, 64],
        [1, 2, 4, 8, 16, 32, 64, 128],
        [2, 4, 8, 16, 32, 64, 128],
        [1, 3, 7, 15, 31, 63, 127],
        [2, 6, 12, 24, 48, 96],
        [10, 20, 40, 80, 160],
        [1, 2, 5, 10, 20, 40, 80, 160],
        [50, 100, 150],
        [40, 80, 120, 160],
        [30, 60, 90, 120, 150, 180, 210, 240],
    ]
    
    best_result = None
    best_size = 0
    
    for start in start_strategies:
        if time.monotonic() > deadline - 5:
            break
        # Verify start is valid
        if not _is_valid(start):
            continue
        
        # Greedy growth
        grown = _grow_greedy(start, deadline, rng)
        if len(grown) > best_size:
            best_size = len(grown)
            best_result = grown
        
        # Local search to try to improve further
        if time.monotonic() < deadline - 10:
            improved = _local_search(grown, deadline, rng)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = improved
    
    # Final attempt: try a few more random starts
    for _ in range(5):
        if time.monotonic() > deadline - 3:
            break
        random_start = [rng.randint(1, 256) for _ in range(rng.randint(1, 5))]
        # Remove duplicates
        random_start = list(set(random_start))
        if not _is_valid(random_start):
            continue
        grown = _grow_greedy(random_start, deadline, rng)
        if len(grown) > best_size:
            best_size = len(grown)
            best_result = grown
    
    if best_result is None:
        # Fallback: return a known valid small set
        return list(range(1, 9))
    
    # Ensure return type is exactly list of int
    return list(best_result)

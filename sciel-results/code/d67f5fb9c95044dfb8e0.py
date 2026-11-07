import time
import random

def _verify(arr):
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
    
    # Bitset DP
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
            
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _is_valid_dp(dp):
    acc = 0
    for mask in dp:
        if mask & acc:
            return False
        acc |= mask
    return True

def _grow(start, deadline):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    
    # Initialize DP for current set
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in cur:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
            
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for v in range(1, 257):
            if v in cur_set:
                continue
            
            # Test adding v
            new_dp = [0] * (n + 2)
            for k in range(n + 1):
                new_dp[k] = dp[k]
            for k in range(n + 1, 0, -1):
                new_dp[k] |= (dp[k - 1] << v)
                
            if _is_valid_dp(new_dp):
                cur.append(v)
                cur_set.add(v)
                dp = new_dp
                improved = True
                break
    return cur

def _local_search(arr, deadline):
    """Try to grow arr by removing one element and adding a different one."""
    cur = list(arr)
    cur_set = set(cur)
    n = len(cur)
    
    # Initialize DP
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in cur:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        for i in range(n):
            if time.monotonic() > deadline:
                break
            removed = cur[i]
            # Remove cur[i]
            cur.pop(i)
            cur_set.discard(removed)
            n -= 1
            
            # Recompute DP without removed element
            dp2 = [0] * (n + 1)
            dp2[0] = 1
            for x in cur:
                for k in range(n, 0, -1):
                    dp2[k] |= (dp2[k - 1] << x)
            
            # Try to add a new element
            for v in range(1, 257):
                if v in cur_set:
                    continue
                
                new_dp = [0] * (n + 2)
                for k in range(n + 1):
                    new_dp[k] = dp2[k]
                for k in range(n + 1, 0, -1):
                    new_dp[k] |= (dp2[k - 1] << v)
                
                if _is_valid_dp(new_dp):
                    cur.append(v)
                    cur_set.add(v)
                    dp = new_dp
                    improved = True
                    break
            
            if not improved:
                # Put removed element back
                cur.insert(i, removed)
                cur_set.add(removed)
                n += 1
                dp = [0] * (n + 1)
                dp[0] = 1
                for x in cur:
                    for k in range(n, 0, -1):
                        dp[k] |= (dp[k - 1] << x)
    
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0
    best = None

    def consider(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            if _verify(arr):
                best = list(arr)

    # 1) Try consecutive runs from high end, largest to smallest
    for n in range(60, 0, -1):
        a = 257 - n
        arr = list(range(a, 257))
        if _verify(arr):
            best = arr
            break
        if time.monotonic() > deadline - 10:
            break

    # 2) Greedy grow from best consecutive
    if best is not None:
        g = _grow(best, deadline)
        if len(g) > len(best):
            best = g

    # 3) Local search: try removing one and adding another
    if best is not None and time.monotonic() < deadline - 5:
        ls = _local_search(best, deadline)
        if len(ls) > len(best):
            best = ls

    # 4) Random restarts with growth, bimodal distribution
    rng = random.Random(12345)
    attempts = 0
    while time.monotonic() < deadline - 3 and attempts < 10000:
        attempts += 1
        n = rng.randint(20, 60)
        # Bimodal: encourage gaps by mixing low and high
        low = list(range(1, 121))
        high = list(range(137, 257))
        half = n // 2
        pool = rng.sample(low, min(half, len(low))) + rng.sample(high, n - min(half, len(low)))
        if len(pool) < n:
            continue
        arr = rng.sample(pool, n)
        if _verify(arr):
            g = _grow(arr, deadline)
            if best is None or len(g) > len(best):
                best = g
            # Try local search on the grown result
            if time.monotonic() < deadline - 3:
                ls = _local_search(g, deadline)
                if len(ls) > len(best):
                    best = ls
            if best is not None and len(best) >= 60:
                break

    # 5) Final fallback: known valid size-31
    if best is None:
        best = list(range(226, 257))

    best = [int(v) for v in best]
    return best

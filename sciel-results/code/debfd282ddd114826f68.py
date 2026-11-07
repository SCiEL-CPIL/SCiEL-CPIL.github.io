import time
import random

def _init_dp(arr):
    n = len(arr)
    if n == 0:
        return [1]
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _is_valid(arr):
    """
    Checks if the set 'arr' satisfies the cardinality separation constraint.
    """
    n = len(arr)
    if n == 0:
        return True
    dp = _init_dp(arr)
    acc = 0
    for k in range(n + 1):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _verify_add(dp, n, x):
    """
    Checks if adding x to a set of size n (represented by dp) is valid.
    Returns new_dp if valid, None otherwise.
    """
    new_dp = [0] * (n + 2)
    for k in range(n + 1):
        new_dp[k] = dp[k]
    
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << x)
        
    acc = 0
    for k in range(n + 2):
        if new_dp[k] & acc:
            return None
        acc |= new_dp[k]
        
    return new_dp

def _grow_greedy(start, strategy='smallest', deadline=None, rng=None):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    dp = _init_dp(cur)
    
    improved = True
    while improved:
        if deadline is not None and time.monotonic() > deadline:
            break
        improved = False
        
        # Get candidates
        candidates = [v for v in range(1, 257) if v not in cur_set]
        if not candidates:
            break
            
        if strategy == 'smallest':
            order = candidates
        elif strategy == 'largest':
            order = list(reversed(candidates))
        elif strategy == 'random':
            order = candidates[:]
            if rng:
                rng.shuffle(order)
            else:
                random.shuffle(order)
        else:
            order = candidates
            
        for v in order:
            new_dp = _verify_add(dp, n, v)
            if new_dp is not None:
                cur.append(v)
                cur_set.add(v)
                n += 1
                dp = new_dp
                improved = True
                break
                
    return cur

def _search_consecutive(deadline):
    """
    Searches for the largest valid set of consecutive integers in 1..256.
    """
    # Max possible size is 60, but range is 256, so max n is 256.
    # However, the problem statement says "1..60 distinct integers".
    # Wait, "construct() -> 1..60 distinct integers".
    # So max size is 60.
    
    for n in range(60, 0, -1):
        # Check if any interval of length n works
        # a can be from 1 to 256 - n + 1
        # We want to find if ANY works.
        # Usually, larger 'a' (shifting right) helps separation?
        # Or smaller 'a'?
        # Let's check a few candidates.
        # To be safe and fast, we can check a=1, a=256-n+1, and a few in between.
        # But actually, we can just check all a if it's fast enough.
        # n=60, a has 197 values. 60 * 197 = ~12000 checks.
        # Each check is O(n * max_sum) bit ops.
        # max_sum for n=60, val=256 is 15360.
        # 60 * 15360 / 64 (word size) ~ 14k ops per check.
        # 12000 * 14k = 168M ops. Might be too slow in Python.
        
        # Optimization:
        # The condition for consecutive integers [a, a+n-1] to be valid
        # is that min(S_k) > max(S_{k-1}) for all k.
        # min(S_k) = k*a + k*(k-1)/2
        # max(S_{k-1}) = (k-1)*(a+n-1) - (k-1)*(k-2)/2
        # We need:
        # k*a + k*(k-1)/2 > (k-1)*(a+n-1) - (k-1)*(k-2)/2
        # k*a + (k^2-k)/2 > (k-1)*a + (k-1)(n-1) - (k^2-3k+2)/2
        # k*a - (k-1)*a + (k^2-k)/2 + (k^2-3k+2)/2 > (k-1)(n-1)
        # a + (2k^2 - 4k + 2)/2 > (k-1)(n-1)
        # a + (k-1)^2 > (k-1)(n-1)
        # a > (k-1)(n-1) - (k-1)^2
        # a > (k-1)(n-1 - (k-1))
        # a > (k-1)(n-k)
        
        # So the condition is: for all k in 1..n, a > (k-1)(n-k).
        # Let f(k) = (k-1)(n-k). We need a > max_{k} f(k).
        # f(k) is a downward parabola. Max is at k = (n+1)/2.
        # So we just need to check k = floor((n+1)/2) and ceil((n+1)/2).
        
        max_k = (n + 1) // 2
        min_required_a = 0
        for k in [max_k, max_k + 1]:
            if 1 <= k <= n:
                val = (k - 1) * (n - k)
                if val > min_required_a:
                    min_required_a = val
        
        # We need a > min_required_a.
        # So a_min = min_required_a + 1.
        # We also need a + n - 1 <= 256 => a <= 256 - n + 1.
        
        a_min = min_required_a + 1
        a_max = 256 - n + 1
        
        if a_min <= a_max:
            # We can pick a = a_min (or any valid a).
            # Let's pick a_min to keep numbers small? Or a_max to keep them large?
            # It doesn't matter for validity, only for the specific values returned.
            # The task asks to "Return a list of ints".
            # Any valid set is fine.
            # Let's return the set starting at a_min.
            return list(range(a_min, a_min + n))
            
    return None

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0  # Safety margin
    best = None
    
    def update_best(arr):
        nonlocal best
        if best is None or len(arr) > len(best):
            best = list(arr)
            
    # 1. Try the analytical consecutive integer search first.
    # This is very fast and likely finds the optimal solution.
    res = _search_consecutive(deadline)
    if res:
        update_best(res)
        
    # If we found a large set (e.g., > 30), we might still want to check if a larger non-consecutive set exists?
    # Unlikely, but let's run the greedy if we have time and the set is small.
    # Actually, if the analytical solution gives n, is it possible to have n+1?
    # The analytical derivation assumes consecutive integers.
    # It's possible a non-consecutive set of size n+1 exists.
    # But given the constraints, the consecutive solution is usually optimal or near-optimal.
    
    # 2. Run Greedy searches to see if we can beat the consecutive result.
    # If the consecutive result is small, greedy might find something better.
    # If it's large (e.g., 30+), greedy is unlikely to beat it quickly.
    
    if best is None or len(best) < 40:
        # Deterministic Greedies
        res = _grow_greedy([], 'smallest', deadline)
        update_best(res)
        
        res = _grow_greedy([], 'largest', deadline)
        update_best(res)
        
        # Start from Powers of 2
        p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        res = _grow_greedy(p2, 'smallest', deadline)
        update_best(res)
        
        # Randomized Restarts
        rng = random.Random(42)
        if time.monotonic() < deadline - 5:
            for _ in range(50):
                if time.monotonic() > deadline - 2:
                    break
                k = rng.randint(0, 5)
                if k == 0:
                    start = []
                else:
                    start = rng.sample(range(1, 257), k)
                    # Quick validity check for start
                    if not _is_valid(start):
                        continue
                strategy = rng.choice(['smallest', 'largest', 'random'])
                res = _grow_greedy(start, strategy, deadline, rng)
                update_best(res)
                
    # Final check: ensure best is valid (paranoia)
    if best is not None:
        if not _is_valid(best):
            # Fallback to a known valid set if something went wrong
            best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            
    if best is None:
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best

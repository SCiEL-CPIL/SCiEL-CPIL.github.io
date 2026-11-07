import time
import random

def _build_dp(arr):
    n = len(arr)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in arr:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
    return dp

def _check_dp(dp):
    acc = 0
    for k in range(len(dp)):
        if dp[k] & acc:
            return False
        acc |= dp[k]
    return True

def _try_add(dp, v, n):
    """
    Try to add v to a set represented by dp (for n elements).
    Returns new_dp if valid, None otherwise.
    """
    new_dp = [0] * (n + 2)
    new_dp[0] = 1
    
    # Copy old parts
    for k in range(n + 1):
        new_dp[k] = dp[k]
        
    # Add contributions from v
    # new_dp[k] |= (dp[k-1] << v)
    # We iterate high to low to ensure we don't use v multiple times in the same step?
    # Actually, new_dp is built from dp. dp is fixed. So order doesn't matter for correctness of OR-ing into new_dp.
    # But wait, new_dp[k] already has dp[k]. We are adding subsets that INCLUDE v.
    # Subsets of size k including v correspond to subsets of size k-1 from the original set (dp[k-1]).
    # So new_dp[k] = dp[k] | (dp[k-1] << v).
    # This is correct.
    
    for k in range(n + 1, 0, -1):
        new_dp[k] |= (dp[k - 1] << v)
        
    # Check validity: no sum should appear in two different cardinality layers
    acc = 0
    for k in range(n + 2):
        if new_dp[k] & acc:
            return None
        acc |= new_dp[k]
        
    return new_dp

def _grow(start, deadline):
    cur = list(start)
    cur_set = set(cur)
    n = len(cur)
    dp = _build_dp(cur)
    
    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        n = len(cur)
        
        # STRUCTURAL CHANGE: Iterate from high to low (256 down to 1)
        # This prioritizes large values which create larger gaps in sums,
        # allowing more elements to be packed.
        for v in range(256, 0, -1):
            if v in cur_set:
                continue
            new_dp = _try_add(dp, v, n)
            if new_dp is not None:
                cur.append(v)
                cur_set.add(v)
                dp = new_dp
                improved = True
                break
                
    return cur

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0  # Safety margin for 90s limit
    best = None
    best_len = 0
    
    def consider(arr):
        nonlocal best, best_len
        if len(arr) > best_len:
            # Verify basic constraints
            if _check_dp(_build_dp(arr)):
                best = list(arr)
                best_len = len(arr)

    # 1. Start with powers of 2 (size 8)
    start_powers = [1, 2, 4, 8, 16, 32, 64, 128]
    g1 = _grow(start_powers, deadline)
    consider(g1)
    
    # 2. Try other structured starts
    structured_starts = [
        [128, 129, 130, 131, 132, 133, 134, 135], # High consecutive
        [10, 20, 40, 80, 160], # Sparse powers
        [50, 100, 150, 200, 250], # Spread out
        [1, 10, 50, 100, 200], # Mixed
    ]
    
    for start in structured_starts:
        if time.monotonic() > deadline - 10:
            break
        if _check_dp(_build_dp(start)):
            g = _grow(start, deadline)
            consider(g)
            
    # 3. Random restarts with high-to-low growth
    rng = random.Random(42)
    attempts = 0
    while time.monotonic() < deadline - 5 and attempts < 1000:
        attempts += 1
        n_start = rng.randint(5, 10)
        pool = list(range(1, 257))
        start = rng.sample(pool, n_start)
        
        if _check_dp(_build_dp(start)):
            g = _grow(start, deadline)
            consider(g)
            
    if best is None:
        best = start_powers
        
    return best

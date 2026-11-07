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
    # Max sum is 60 * 256 = 15360
    # We use a list of ints as bitsets
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

def construct():
    t0 = time.monotonic()
    deadline = t0 + 85.0
    best = list(range(226, 257)) # Size 31, known valid
    
    # Try to find size 32
    # Strategy: Random subsets of size 32 from 1..256
    # Bias towards "tight" clusters in the upper half?
    # Or just random?
    
    rng = random.Random(42)
    
    # First, try some structured sets of size 32
    # 1. Consecutive 32 from high end? Fails.
    # 2. Consecutive 32 from middle? Fails.
    # 3. "Gappy" sets.
    
    attempts = 0
    while time.monotonic() < deadline and attempts < 50000:
        attempts += 1
        # Generate a random set of size 32
        # Try different ranges
        # Range 1: 1..256
        # Range 2: 100..256
        # Range 3: 150..256
        
        r = rng.randint(0, 2)
        if r == 0:
            pool = list(range(1, 257))
        elif r == 1:
            pool = list(range(100, 257))
        else:
            pool = list(range(150, 257))
            
        if len(pool) < 32:
            continue
            
        cand = rng.sample(pool, 32)
        if _verify(cand):
            best = cand
            break # Found size 32
            
    # If no 32 found, try to grow best (31) to 32?
    # The current best is likely 31.
    # Try to add one element to best.
    if len(best) < 32:
        cur = list(best)
        cur_set = set(cur)
        for v in range(1, 257):
            if v in cur_set:
                continue
            cand = cur + [v]
            if _verify(cand):
                best = cand
                break

    return best

import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # Sieve for sigma: sigma[n] = sum of divisors of n
    # For each d from 1 to MAX_N, add d to sigma[m] for all m = d, 2d, 3d, ...
    sigma = [0] * (MAX_N + 1)
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d
    
    best_error = float('inf')
    best_a = None
    best_b = None
    
    # Search for a/b close to 3/2
    # a ≈ 3k, b ≈ 2k
    # Check small offsets d_a, d_b around 3k and 2k
    # We need a, b in [1, MAX_N]
    
    # For efficiency, check k values and small offsets
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # To minimize this, we want |2a - 3b| small and b large
    
    # Strategy: iterate over b from MAX_N down to 1 (larger b gives smaller denominator)
    # For each b, a should be close to 3b/2
    # Check a in a small window around 3b/2
    
    # But checking all b is too slow. Instead, iterate over k and check offsets.
    
    # Let's try: for k from 1 to MAX_N//2, a = 3k + da, b = 2k + db
    # where da, db are small integers
    # |a/b - 3/2| = |(3k+da)/(2k+db) - 3/2| = |2(3k+da) - 3(2k+db)| / (2(2k+db))
    # = |6k + 2da - 6k - 3db| / (2(2k+db)) = |2da - 3db| / (2(2k+db))
    
    # So for fixed da, db, the error is |2da - 3db| / (2(2k+db))
    # To minimize, we want large k and small |2da - 3db|
    
    # The minimum nonzero |2da - 3db| is 1 (e.g., da=1, db=1: |2-3|=1)
    # So minimum possible error for given k is 1/(2(2k+db))
    
    # Let's check da, db in range [-10, 10] for each k, but that's 200 * 5M = 1B checks
    # Too slow. Instead, for each k, check a few promising (da, db) pairs.
    
    # Actually, let's be smarter. For each k, the best (da, db) minimizes |2da - 3db| / (2k + db)
    # subject to 1 <= 3k+da <= MAX_N and 1 <= 2k+db <= MAX_N
    
    # The minimum |2da - 3db| for nonzero is 1. So we want pairs with |2da - 3db| = 1.
    # Solutions: da = 2t+1, db = 2t for any integer t? No.
    # 2da - 3db = 1: da = (3db+1)/2, so db must be odd. db=1: da=2; db=3: da=5; etc.
    # 2da - 3db = -1: da = (3db-1)/2, db odd. db=1: da=1; db=3: da=4; etc.
    
    # So promising pairs have |2da - 3db| = 1, giving error = 1/(2(2k+db))
    
    # For each k, try several (da, db) with |2da-3db|=1, check bounds, then check sigma equality.
    
    # Precompute: for db in range, da = (3*db + 1)//2 if (3*db+1) even, or (3*db-1)//2
    # Let's just enumerate small da, db and filter by |2da-3db| being small.
    
    # Generate candidate (da, db) pairs with |2da - 3db| <= 5
    candidates = []
    for da in range(-20, 21):
        for db in range(-20, 21):
            val = abs(2*da - 3*db)
            if val > 0 and val <= 5:  # nonzero error, small
                candidates.append((da, db, val))
    
    # Sort by val (error numerator) ascending, then by |db| ascending (prefer smaller |db| for larger 2k+db)
    candidates.sort(key=lambda x: (x[2], abs(x[1])))
    
    # Now iterate k from large to small (larger k gives smaller error)
    # For each k, try candidates in order, check bounds and sigma equality
    
    # To limit work, cap the number of k values we check
    # Error for best case ~ 1/(4k), so for k=1e6, error ~ 2.5e-7
    # Current best from seed: |11/6 - 3/2| = |11/6 - 9/6| = 2/6 = 1/3 ≈ 0.333
    # We can do much better.
    
    # Iterate k from MAX_N//2 down to 1, but stop when even best possible error exceeds current best
    # Best possible error for k is 1/(2*(2k + min_db)) where min_db is smallest |db| in candidates
    # Actually, just iterate and break when 1/(4k) > best_error (rough upper bound)
    
    k_max = MAX_N // 2  # 5,000,000
    k_min = 1
    
    # Iterate from large k to small k
    for k in range(k_max, k_min, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Check if even the best possible error for this k is better than current best
        # Best error ~ 1/(2*(2k)) = 1/(4k) for the smallest numerator
        # If 1/(4k) >= best_error, then no k' <= k can beat it (since error increases as k decreases)
        # Actually error = val/(2*(2k+db)), for fixed val, decreases as k increases
        # So we should go from large k to small k, and once 1/(4k) > best_error, we can stop
        # But db can be negative, so 2k+db could be smaller. Let's just not break early for safety.
        
        for da, db, val in candidates:
            a = 3 * k + da
            b = 2 * k + db
            
            if a < 1 or a > MAX_N or b < 1 or b > MAX_N:
                continue
            
            # Check sigma equality
            if sigma[a] == sigma[b]:
                # Compute error
                # |a/b - 3/2| = |2a - 3b| / (2b) = val / (2*b)
                error = val / (2 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
        
        # Early termination: if even the best possible error for smaller k is worse
        # For k-1, best error ~ 1/(4*(k-1)) which is larger than 1/(4k)
        # So once 1/(4k) > best_error, we can break
        if 1.0 / (4.0 * k) > best_error:
            break
    
    if best_a is None:
        # Fallback to seed
        return [11, 6]
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END

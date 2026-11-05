import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Try a large limit first
    LIMIT = 10**10
    
    try:
        # 1. Generate powerful numbers
        powerful_set = set()
        max_v = int(LIMIT ** (1/3)) + 1
        
        # Precompute square-free check for small v
        # v <= 2154
        # Primes up to sqrt(2154) ~ 46
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        
        def is_square_free(n):
            for p in primes:
                if p*p > n: break
                if n % (p*p) == 0:
                    return False
            return True

        for v in range(1, max_v + 1):
            if not is_square_free(v):
                continue
            v3 = v ** 3
            if v3 > LIMIT:
                break
            max_u = isqrt(LIMIT // v3)
            for u in range(1, max_u + 1):
                powerful_set.add(u * u * v3)
                
        P = sorted(powerful_set)
        K = len(P)
        
        # Check time
        if time.time() > deadline - 5:
            # Fallback to smaller limit
            return construct_small()

        # 2. Count sums
        sum_counts = {}
        # Use local variables for speed
        P_local = P
        K_local = K
        
        for i in range(K_local):
            a = P_local[i]
            # Optimization: if a + P_local[i] > some limit? No, we want max count.
            for j in range(i, K_local):
                s = a + P_local[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                
        # Check time
        if time.time() > deadline - 5:
            # We have counts, but maybe not all? No, loop finished.
            # Just proceed to find max.
            pass

        # 3. Find best N
        best_N = 0
        max_count = 0
        for s, c in sum_counts.items():
            if c > max_count:
                max_count = c
                best_N = s
                
        # 4. Reconstruct pairs
        P_set = set(P)
        pairs = []
        for a in P:
            if a > best_N // 2:
                break
            b = best_N - a
            if b in P_set:
                pairs.append([a, b])
                
        return {'N': best_N, 'pairs': pairs}
        
    except Exception:
        return construct_small()

def construct_small():
    # Fallback to a very safe, fast limit
    limit = 20000
    powerful = sorted({a * a * b ** 3 for b in range(1, 28) for a in range(1, isqrt(limit // b ** 3) + 1)})
    members = set(powerful)
    best = max(range(2, limit + 1), key=lambda n: sum(1 for a in powerful if 2 * a <= n and n - a in members))
    return {'N': best, 'pairs': [[a, best - a] for a in powerful if 2 * a <= best and best - a in members]}

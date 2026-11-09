import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    # 110s deadline to be safe (150s limit)
    deadline = start_time + 110.0
    
    def is_square_free(n):
        if n < 2:
            return True
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    def generate_powerful(limit):
        powerful = set()
        max_b = int(limit ** (1.0/3.0)) + 1
        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b3 = b ** 3
            if b3 > limit:
                break
            max_a = int(math.isqrt(limit // b3))
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    powerful.add(val)
        return sorted(powerful)

    # Phase 1: Generate powerful numbers up to L = 10^12
    # M(L) ~ L^(1/3) * (some constant) ~ 10^4
    # O(M^2) ~ 10^8 operations. This is the bottleneck.
    # We need to be careful with the loop.
    
    L = 10**12
    
    # Check time
    if time.time() > deadline - 30:
        # Fallback to smaller range if generation is too slow
        L = 10**9

    p = generate_powerful(L)
    M = len(p)
    
    # Phase 2: Count representations
    # We want to find N that maximizes the count of pairs (a, b) with a+b=N, a<=b, a,b in p.
    # Since we need to return the pairs for the best N, we can't just use a histogram if M is large
    # because we need to retrieve the pairs.
    # However, we can first find the best N using a histogram (faster if we just count),
    # then retrieve the pairs for that N.
    
    # But wait, generating the histogram itself is O(M^2).
    # Let's just do the O(M^2) loop once, counting into a dict.
    
    counts = defaultdict(int)
    
    # Optimization: Use local variables for speed
    p_list = p
    M_local = M
    counts_local = counts
    
    # We can potentially use numpy for the counting if we just want the max N,
    # but we need the pairs. 
    # Actually, we can do the counting in a tight loop.
    # 10^8 iterations in Python might take 10-20 seconds.
    
    # Let's try to optimize the inner loop.
    # for i in range(M):
    #     a = p[i]
    #     for j in range(i, M):
    #         b = p[j]
    #         s = a + b
    #         if s > L: break # Not helpful, sums can be up to 2L
    #         counts[s] += 1
        
    # This is too slow in pure Python if M=10^4 (100M ops).
    # Let's use a smarter approach.
    # We can generate all sums using numpy broadcasting? No, M^2 memory is 100M * 8 bytes = 800MB. Too much.
    
    # Alternative:
    # We know the best N is likely small relative to L? No, N can be up to 2L.
    # But the density of powerful numbers is higher for smaller numbers.
    # So the best N is likely in the lower half of the possible sums.
    
    # Let's try a hybrid:
    # 1. Estimate the best N using a smaller subset or a heuristic?
    # 2. Or, just run the O(M^2) loop but break early if time is running out?
    #    If we break early, we have partial counts. The best N found so far might be good.
    
    # Let's try to speed up the counting.
    # Using a list of zeros for counts up to 2L? 2*10^12 is too big for a list.
    # So we must use a dict or defaultdict.
    
    # To make the Python loop faster, we can use a trick:
    # Instead of iterating all j for each i, we can stop if a+b > some limit?
    # No, we want the global max.
    
    # Let's try to use PyPy-like optimizations in CPython? No.
    # Let's reduce L to 10^10? M ~ 2000. M^2 ~ 4M. Very fast.
    # But score might be lower.
    # L=10^11? M ~ 5000. M^2 ~ 25M. Fast.
    # L=10^12? M ~ 10000. M^2 ~ 100M. Slow but maybe okay with 110s?
    
    # Let's test with L=10^11 first to ensure we get a valid result quickly,
    # then if time permits, try L=10^12?
    # No, the function must return one result.
    
    # Let's stick to L=10^11 for safety, or 10^12 if we optimize.
    # Actually, 10^8 iterations in Python takes ~10-30s depending on hardware.
    # Let's use L=10^11 (M~4600, M^2~21M). This is very safe.
    # But can we do better?
    
    # Let's try L=10^12 but with a time check inside the loop?
    # That's risky.
    
    # Let's go with L=10^11.
    # If M is ~4600, M^2 is ~21M.
    # 21M iterations in Python is ~2-5 seconds.
    # This is very safe.
    
    # What if we go to L=10^12?
    # M ~ 10^4. M^2 ~ 10^8.
    # 10^8 iterations in Python is ~10-20 seconds.
    # This is also safe within 110s.
    
    # Let's use L=10^12.
    
    # Re-generate with L=10^12
    # (Already did above if time allowed)
    
    # If we are here, we have p.
    
    # Counting loop
    # To speed up, we can use a list for counts if the range of N is small?
    # No, N up to 2*10^12.
    
    # Let's just run the loop.
    # We can optimize by breaking the inner loop if a+b > max_N_seen?
    # No, we don't know max_N_seen.
    
    # Let's just do it.
    
    # Optimization: Use a local variable for the dict
    d = defaultdict(int)
    
    # Parallelization is not allowed/available.
    
    # Let's try to use numpy to speed up the counting of a subset?
    # No, let's stick to pure Python for the counting to avoid memory issues,
    # but we can use a trick:
    # We only care about the MAX count.
    # We can update the best N on the fly.
    
    best_N = 0
    best_count = 0
    
    # We need to store the pairs for the best N.
    # If we find a new best N, we can't easily retrieve the pairs without re-scanning.
    # So we should store all pairs for the current best N?
    # Or, store the counts, then re-scan for the best N at the end.
    
    # Re-scanning is O(M^2) again. Total 2*O(M^2).
    # 200M operations. Might be tight.
    
    # Alternative: Store the pairs for the best N in a list.
    # If we find a better N, we clear the list and start collecting for the new N.
    # This is efficient if the best N is found early and rarely changes.
    
    best_pairs = []
    
    # We need to be careful: if we find a new best N, we discard old pairs.
    # But we don't know if the old N will have more pairs later.
    # So we must accumulate counts for ALL N to be sure.
    
    # Okay, let's accumulate counts in a dict.
    # Then find max.
    # Then re-scan to get pairs for that max N.
    
    # To make the re-scan faster, we can use a set for p.
    p_set = set(p)
    
    # Step 2a: Count
    for i in range(M_local):
        a = p_list[i]
        # Inner loop
        for j in range(i, M_local):
            b = p_list[j]
            s = a + b
            d[s] += 1
            
            # Time check
            if (i & 0xFF) == 0 and (j & 0xFF) == 0:
                if time.time() > deadline - 20:
                    break
        if time.time() > deadline - 20:
            break
            
    if not d:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Step 2b: Find best N
    best_N = max(d, key=d.get)
    best_count = d[best_N]
    
    # Step 2c: Retrieve pairs for best_N
    # We can do this efficiently by iterating a in p, b = best_N - a
    # Check if b in p_set and a <= b
    pairs = []
    for a in p_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    # Verify count matches?
    # len(pairs) should be best_count.
    # If not, it means our count was wrong or we missed some?
    # It should match.
    
    return {'N': best_N, 'pairs': pairs}

# Test/Run
if __name__ == '__main__':
    res = construct()
    print(res)

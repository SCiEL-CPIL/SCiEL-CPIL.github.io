import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s
    
    N = 10_000_000
    
    # Step 1: Compute smallest prime factor (SPF) up to N
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Check time
    if time.time() > deadline:
        return [11, 6]
    
    # Step 2: Compute sigma(n) for all n using SPF
    sigma = [0] * (N + 1)
    sigma[1] = 1
    for n in range(2, N + 1):
        # Factorize n using SPF
        m = n
        prod = 1
        while m > 1:
            p = spf[m]
            # Compute p^e and sum p^0 + p^1 + ... + p^e
            pe = 1
            s = 1
            while m % p == 0:
                m //= p
                pe *= p
                s += pe
            prod *= s
        sigma[n] = prod
    
    # Check time
    if time.time() > deadline:
        return [11, 6]
    
    # Step 3: Group numbers by sigma value
    # To save memory, only keep groups with at least 2 elements
    # Use a dict: sigma_value -> list of n
    # But this could use too much memory. Instead, let's be smarter.
    
    # Alternative: for each n, we want to find m such that sigma(m) = sigma(n) and |n/m - 3/2| is minimized.
    # This is equivalent to: for each n, look for m ≈ 2n/3 (if n > m) or m ≈ 3n/2 (if m > n)
    # with sigma(m) = sigma(n).
    
    # Let's build a dict from sigma value to list of n's, but limit the list size.
    # Actually, let's just build it and see if memory is okay.
    
    groups = {}
    for n in range(1, N + 1):
        s = sigma[n]
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
    
    # Check time
    if time.time() > deadline:
        return [11, 6]
    
    # Step 4: For each group, find the pair minimizing |a/b - 3/2|
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 3/2)  # = 1/3
    
    for s, lst in groups.items():
        if len(lst) < 2:
            continue
        # Sort the list
        lst.sort()
        # For each b in lst, find a in lst such that a ≈ 3b/2
        # Use two pointers or binary search
        # Since lst is sorted, for each b, the best a is the one closest to 3b/2
        # We can use binary search for each b, but that's O(k log k) per group.
        # Alternatively, use two pointers: for each b, move a pointer.
        
        # Let's use: for each b, target_a = 3*b/2. Find a in lst closest to target_a.
        # Since lst is sorted, we can do a binary search.
        
        k = len(lst)
        for i in range(k):
            b = lst[i]
            target_a = 3 * b // 2  # approximate
            # Binary search for target_a in lst
            # We want a such that a/b ≈ 3/2, so a ≈ 3b/2
            # Also consider a/b and b/a, but we want |a/b - 3/2|, so a should be larger than b typically.
            # Actually, we can have a < b too, but then a/b < 1, and |a/b - 3/2| > 1/2, which is worse than 1/3.
            # So we only need a > b, i.e., a ≈ 3b/2 > b.
            
            # Find the position in lst where target_a would be
            lo, hi = 0, k - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if lst[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            for pos in [lo, lo - 1]:
                if 0 <= pos < k:
                    a = lst[pos]
                    if a == b:
                        continue
                    err = abs(a / b - 3 / 2)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
    
    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END

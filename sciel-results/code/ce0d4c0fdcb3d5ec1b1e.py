import math
import time
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    MAX_VAL = 10210
    MIN_VAL = 211  # strictly larger than 210
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def compute_sf(n):
        """Compute square-free part of n."""
        sf_val = 1
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                sf_val *= p
        return sf_val
    
    # Precompute square-free parts for all numbers 211..10210
    sf_cache = {}
    for i in range(MIN_VAL, MAX_VAL + 1):
        sf_cache[i] = compute_sf(i)
    
    # Build reverse map: sf_value -> sorted list of numbers with that sf
    sf_to_nums = defaultdict(list)
    for i in range(MIN_VAL, MAX_VAL + 1):
        sf_to_nums[sf_cache[i]].append(i)
    for v in sf_to_nums:
        sf_to_nums[v].sort()
    
    def sf_xor(sf_a, target):
        """Return sf_c such that sf(sf_a * sf_c) = target."""
        def prime_set(n):
            ps = set()
            x = n
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
                if p * p > x:
                    break
                if x % p == 0:
                    ps.add(p)
                    while x % p == 0:
                        x //= p
            if x > 1:
                ps.add(x)
            return ps
        
        pa = prime_set(sf_a)
        pt = prime_set(target)
        pc = pa ^ pt  # symmetric difference
        result = 1
        for p in pc:
            result *= p
        return result
    
    # 1. Single element: find smallest n > 210 such that 210*n is a perfect square
    # 210 = 2*3*5*7, so we need n = 210 * k^2 for some k
    # Smallest such n > 210: k=1 gives 210 (not > 210), k=2 gives 840
    best_sol = [840]
    best_raw = 630
    
    # 2. Pair search: find a, b > 210, a != b, such that 210*a*b is a perfect square
    # This means sf(210) * sf(a) * sf(b) = 1, i.e., sf(a) * sf(b) = sf(210)
    # sf(210) = 210 (since 210 is square-free)
    # So we need sf(a) * sf(b) = 210, meaning sf(b) = 210 / sf(a) if sf(a) divides 210
    # More generally, sf(a) XOR sf(b) = sf(210) in the sense that the product's sf is 210
    
    for a in range(MIN_VAL, MAX_VAL + 1):
        if a >= best_raw + 210:
            break
        if time.time() > deadline:
            break
        sf_a = sf_cache[a]
        sf_b_needed = sf_xor(sf_a, 210)
        if sf_b_needed in sf_to_nums:
            nums = sf_to_nums[sf_b_needed]
            idx = bisect.bisect_right(nums, a)
            if idx < len(nums):
                b = nums[idx]
                if b > a and b <= MAX_VAL:
                    raw = b - 210
                    if raw < best_raw:
                        best_raw = raw
                        best_sol = [a, b]
    
    # 3. Triplet search: find a, b, c > 210, all distinct, such that 210*a*b*c is a perfect square
    # We iterate M (the maximum element) upward, and for each M, find a, c < M
    # such that sf(a) * sf(c) = sf_xor(sf(M), 210)
    
    for M in range(MIN_VAL, min(best_raw + 210, MAX_VAL + 1)):
        if time.time() > deadline:
            break
        sf_M = sf_cache[M]
        T = sf_xor(sf_M, 210)
        # Need a, c in (210, M), a < c, a != c, sf(a*c) = T
        # For each a, compute sf_c_needed = sf_xor(sf(a), T)
        # and check if there's a c in (a, M) with that sf
        
        for a in range(MIN_VAL, M):
            if time.time() > deadline:
                break
            sf_a = sf_cache[a]
            sf_c_needed = sf_xor(sf_a, T)
            if sf_c_needed in sf_to_nums:
                nums_c = sf_to_nums[sf_c_needed]
                idx = bisect.bisect_right(nums_c, a)
                if idx < len(nums_c) and nums_c[idx] < M:
                    c = nums_c[idx]
                    if a != c and a != M and c != M:
                        # Verify
                        prod = a * c * M * 210
                        k = math.isqrt(prod)
                        if k * k == prod:
                            best_raw = M - 210
                            best_sol = [a, c, M]
                            return [int(a), int(c), int(M)]
    
    # 4. Four-element search if needed - optimized
    if best_raw > 100:
        for M in range(MIN_VAL, min(best_raw + 210, MAX_VAL + 1)):
            if time.time() > deadline:
                break
            sf_M = sf_cache[M]
            T = sf_xor(sf_M, 210)
            seen_sfs_a = set()
            for a in range(MIN_VAL, M):
                if time.time() > deadline:
                    break
                sf_a = sf_cache[a]
                if sf_a in seen_sfs_a:
                    continue
                seen_sfs_a.add(sf_a)
                T_ab = sf_xor(sf_a, T)
                seen_sfs_b = set()
                for b in range(a + 1, M):
                    if time.time() > deadline:
                        break
                    sf_b = sf_cache[b]
                    if sf_b in seen_sfs_b:
                        continue
                    seen_sfs_b.add(sf_b)
                    sf_c_needed = sf_xor(sf_b, T_ab)
                    if sf_c_needed in sf_to_nums:
                        nums_c = sf_to_nums[sf_c_needed]
                        idx = bisect.bisect_right(nums_c, b)
                        if idx < len(nums_c) and nums_c[idx] < M:
                            c = nums_c[idx]
                            if a != c and b != c and a != M and b != M and c != M:
                                prod = a * b * c * M * 210
                                k = math.isqrt(prod)
                                if k * k == prod:
                                    best_raw = M - 210
                                    best_sol = [a, b, c, M]
                                    return [int(a), int(b), int(c), int(M)]
    
    result = [int(x) for x in best_sol]
    return result

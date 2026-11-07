import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin within 90s
    
    N = 10_000_000
    
    # Step 1: Build smallest prime factor sieve (O(N log log N))
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 40:
        return [11, 6]
    
    # Step 2: Compute sigma(n) in O(N) using multiplicative recurrence
    # sigma[n] = sigma[n/p] * (1+p+...+p^k) / (1+p+...+p^(k-1))
    # where p=spf[n] and p^k || n
    # We track p_pow[n] = p^k (highest power of spf[n] dividing n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    p_pow = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    p_pow[1] = 1
    
    for n in range(2, N + 1):
        if time.time() - start_time > deadline - 35:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p^(k+1) | n, so p_pow[m] = p^k
            p_pow[n] = p_pow[m] * p
            # sigma[n] = sigma[m] * (1+p+...+p^(k+1)) / (1+p+...+p^k)
            # = sigma[m] * (p_pow[n]+1) / (p_pow[m]+1)  ... no
            # Actually: sigma[n] = sigma[m/p^k] * (1+p+...+p^(k+1))
            # and sigma[m] = sigma[m/p^k] * (1+p+...+p^k)
            # So sigma[n] = sigma[m] * (1+p+...+p^(k+1)) / (1+p+...+p^k)
            # The ratio (1+p+...+p^(k+1))/(1+p+...+p^k) is not integer in general
            # Better: sigma[n] = sigma[m] // (1+p+...+p^k) * (1+p+...+p^(k+1))
            # But we don't store (1+p+...+p^k) separately...
            # Alternative: just compute the geometric sum
            # 1+p+...+p^(k+1) = (p^(k+1)-1)/(p-1) for p>1
            # 1+p+...+p^k = (p^k-1)/(p-1)
            # sigma[n] = sigma[m] * (p^(k+1)-1) / (p^k-1)
            # This is exact integer division since p^k-1 divides p^(k+1)-1
            # Actually (p^(k+1)-1)/(p^k-1) = (p*p^k - 1)/(p^k - 1) which is NOT always integer!
            # E.g. p=2,k=1: (4-1)/(2-1)=3, sigma[n]=sigma[m]*3... but sigma[4]=7, sigma[2]=3, 3*3=9≠7
            # So this formula is wrong. Let me think again.
            # sigma[n] where n=p^(k+1)*r, gcd(p,r)=1: sigma[n] = (1+p+...+p^(k+1))*sigma[r]
            # sigma[m] where m=p^k*r: sigma[m] = (1+p+...+p^k)*sigma[r]
            # So sigma[n]/sigma[m] = (1+p+...+p^(k+1))/(1+p+...+p^k)
            # For p=2,k=1: (1+2+4)/(1+2) = 7/3, not integer.
            # So we can't just multiply/divide sigma[m].
            # We need to store the geometric sum separately or recompute.
            # Let's store geo[n] = 1+p+...+p^k where p^k || n (highest power of spf[n])
            # Then sigma[n] = sigma[n/(p^k)] * geo[n]
            # And for m = n/p = p^k * r: geo[m] = 1+p+...+p^k = geo[n] - p^(k+1) = geo[n] - p_pow[n]
            # Wait: geo[n] = 1+p+...+p^(k+1), geo[m] = 1+p+...+p^k
            # geo[n] = geo[m] + p^(k+1) = geo[m] + p_pow[n]
            # sigma[n] = sigma[n/p^(k+1)] * geo[n]
            # sigma[m] = sigma[m/p^k] * geo[m] = sigma[n/p^(k+1)] * geo[m]
            # So sigma[n] = sigma[m] * geo[n] / geo[m]
            # geo[n] / geo[m] = (1+p+...+p^(k+1))/(1+p+...+p^k) - still not integer!
            # OK so the cleanest is: sigma[n] = sigma[n // p_pow[n]] * geo[n]
            # where n // p_pow[n] has spf different from p (or is 1)
            # This requires knowing n // p_pow[n], which is expensive to compute in O(1).
            # 
            # Simplest O(N) approach: just use the factorization loop but optimized.
            # Or: use a different recurrence.
            # 
            # Actually, the standard O(N) sigma sieve is:
            # Initialize sigma[n] = 1 for all n
            # For each prime p, for each multiple m of p:
            #   if p^2 does not divide m: sigma[m] *= (1+p)
            #   else: this gets complex
            #
            # Let me just use the per-number factorization but with numpy vectorization where possible.
            # Given the time constraints, let me try a simpler approach:
            # Compute sigma using the divisor-sum sieve: sigma[n] = sum of d|n d
            # This is O(N log N) which might be too slow for N=10^7.
            #
            # Let me try the per-number factorization but break early if too slow.
            pass
        else:
            # p || n (p^1), so p_pow[n] = p
            p_pow[n] = p
            # n = p * m, gcd(p,m)=1
            # sigma[n] = sigma[m] * (1+p)
            sigma[n] = sigma[m] * (1 + p)
    
    # The above only handles the case where p || n. For p^k || n with k>1,
    # we need a different approach. Let me redo with a proper method.
    # 
    # Proper O(N) method:
    # Let lp[n] = largest prime factor of n (or spf).
    # Let n = p^k * m where p = spf[n], gcd(p,m) = 1.
    # Define:
    #   p_pow[n] = p^k
    #   p_sum[n] = 1 + p + p^2 + ... + p^k
    # Then:
    #   sigma[n] = sigma[m] * p_sum[n]  where m = n / p^k
    #
    # Recurrence:
    #   If spf[n] = p and (n/p) % p != 0: k=1, p_pow[n]=p, p_sum[n]=1+p, m=n/p
    #     sigma[n] = sigma[n/p] * (1+p)
    #   If spf[n] = p and (n/p) % p == 0: let m' = n/p. Then spf[m']=p, p_pow[m']=p^(k-1)
    #     p_pow[n] = p_pow[m'] * p
    #     p_sum[n] = p_sum[m'] + p_pow[n]
    #     m = n / p^k = m' / p^(k-1)
    #     sigma[n] = sigma[m] * p_sum[n]
    #     But sigma[m'] = sigma[m] * p_sum[m']
    #     So sigma[m] = sigma[m'] / p_sum[m']  -- NOT integer division in general!
    #
    # Hmm, this is tricky. The issue is that sigma[m'] / p_sum[m'] = sigma[m] which IS an integer,
    # but we'd need to do integer division. Since sigma[m'] = sigma[m] * p_sum[m'] exactly,
    # sigma[m'] % p_sum[m'] == 0, so integer division works!
    #
    # So: sigma[n] = (sigma[m'] // p_sum[m']) * p_sum[n]
    #
    # Let me implement this properly.
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    p_pow = np.zeros(N + 1, dtype=np.int64)
    p_sum = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    p_pow[1] = 1
    p_sum[1] = 1
    
    for n in range(2, N + 1):
        if time.time() - start_time > deadline - 35:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p^(k+1) | n, m = p^k * r
            p_pow[n] = p_pow[m] * p
            p_sum[n] = p_sum[m] + p_pow[n]
            # sigma[n] = sigma[m] // p_sum[m] * p_sum[n]
            sigma[n] = (sigma[m] // p_sum[m]) * p_sum[n]
        else:
            # p || n
            p_pow[n] = p
            p_sum[n] = 1 + p
            sigma[n] = sigma[m] * p_sum[n]
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return [11, 6]
    
    # Step 3: Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        if time.time() - start_time > deadline - 15:
            break
        s = int(sigma[n])
        groups[s].append(n)
    
    # Step 4: Search for best pair
    group_list = [(len(v), k, v) for k, v in groups.items() if len(v) >= 2]
    group_list.sort(key=lambda x: -x[0])
    
    best_num = 0
    best_den = 0
    best_a = 11
    best_b = 6
    
    for size, s_val, nums in group_list:
        if time.time() - start_time > deadline - 5:
            break
        
        nums.sort()
        n_len = len(nums)
        
        for i in range(n_len):
            b = nums[i]
            target = (3 * b) // 2
            lo = 0
            hi = n_len
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            for j in [lo, lo - 1]:
                if 0 <= j < n_len and j != i:
                    a = nums[j]
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue
                    if best_den == 0 or num * best_den < best_num * den:
                        best_num = num
                        best_den = den
                        best_a = a
                        best_b = b
    
    return [int(best_a), int(best_b)]

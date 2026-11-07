import time
import bisect

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill

    # Linear sieve for sigma(n)
    # sigma is multiplicative. For prime p: sigma(p) = p+1
    # For p^k: sigma(p^k) = 1 + p + ... + p^k
    # We track: spf (smallest prime factor), ppow (highest power of spf dividing n), 
    # and sigma_val for each n.
    
    sigma = [0] * (N + 1)
    sigma[1] = 1
    is_composite = [False] * (N + 1)
    primes = []
    # ppow[i] = highest power of spf(i) that divides i
    ppow = [0] * (N + 1)
    ppow[1] = 1
    # sigma_pow[i] = sigma(ppow[i])
    sigma_pow = [0] * (N + 1)
    sigma_pow[1] = 1

    for i in range(2, N + 1):
        if time.time() > deadline:
            break
        if not is_composite[i]:
            primes.append(i)
            ppow[i] = i
            sigma_pow[i] = 1 + i
            sigma[i] = 1 + i
        for p in primes:
            if i * p > N:
                break
            is_composite[i * p] = True
            if i % p == 0:
                # p is the smallest prime factor of i, so spf(i*p) = p
                # ppow(i*p) = ppow(i) * p
                ppow[i * p] = ppow[i] * p
                # sigma(ppow(i*p)) = sigma(ppow(i)) + ppow(i)*p
                sigma_pow[i * p] = sigma_pow[i] + ppow[i] * p
                # sigma(i*p) = sigma(i / ppow(i)) * sigma(ppow(i*p))
                # sigma(i) = sigma(i / ppow(i)) * sigma(ppow(i))
                # So sigma(i*p) = sigma(i) / sigma(ppow(i)) * sigma(ppow(i*p))
                # But we need integer arithmetic. 
                # sigma(i) = sigma(i // ppow[i]) * sigma_pow[i]
                # sigma(i*p) = sigma(i // ppow[i]) * sigma_pow[i*p]
                sigma[i * p] = (sigma[i] // sigma_pow[i]) * sigma_pow[i * p]
                break
            else:
                # p does not divide i, so spf(i*p) = p, ppow(i*p) = p
                ppow[i * p] = p
                sigma_pow[i * p] = 1 + p
                # sigma(i*p) = sigma(i) * sigma(p) = sigma(i) * (1+p)
                sigma[i * p] = sigma[i] * (1 + p)

    if time.time() > deadline:
        # Fallback: return a known valid pair
        # sigma(11)=12, sigma(6)=12, |11/6 - 1.5| = |1.8333-1.5| = 0.3333
        return [11, 6]

    # Group indices by sigma value
    # Use a dict: sigma_value -> list of n
    groups = {}
    for i in range(1, N + 1):
        if time.time() > deadline:
            break
        s = sigma[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]

    if time.time() > deadline:
        return [11, 6]

    # Find best pair
    best_err = 1.0 / 3.0  # Seed error: |11/6 - 3/2| = |11-9|/6 = 2/6 = 1/3
    best_a, best_b = 11, 6

    for s, nums in groups.items():
        if time.time() > deadline:
            break
        if len(nums) < 2:
            continue

        nums.sort()
        K = len(nums)

        if K <= 100:
            # Brute force all pairs
            for i in range(K):
                a = nums[i]
                for j in range(K):
                    b = nums[j]
                    if a == b:
                        continue
                    # Quick prune: |a/b - 1.5| < best_err
                    # a/b in (1.5 - best_err, 1.5 + best_err)
                    # a in b*(1.5 - best_err), b*(1.5 + best_err)
                    lo = b * (1.5 - best_err)
                    hi = b * (1.5 + best_err)
                    if a < lo or a > hi:
                        continue
                    # Exact rational comparison: |a/b - 3/2| = |2a - 3b| / (2b)
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        # This would be perfect but task says raw value 0 is invalid
                        continue
                    err = num / (2 * b)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b
        else:
            # Binary search approach
            for a in nums:
                # We want b such that a/b ≈ 3/2, i.e., b ≈ 2a/3
                target_b = 2 * a // 3
                # Find b in nums closest to target_b
                pos = bisect.bisect_left(nums, target_b)
                candidates = []
                if pos < K:
                    candidates.append(nums[pos])
                if pos > 0:
                    candidates.append(nums[pos - 1])
                # Also check nearby positions for safety
                for delta in range(-2, 3):
                    idx = pos + delta
                    if 0 <= idx < K:
                        if nums[idx] not in candidates:
                            candidates.append(nums[idx])

                for b in candidates:
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    err = num / (2 * b)
                    if err < best_err:
                        best_err = err
                        best_a, best_b = a, b

    return [int(best_a), int(best_b)]

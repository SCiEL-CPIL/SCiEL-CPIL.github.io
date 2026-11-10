import numpy as np
import time

def construct():
    N = 10_000_000
    deadline = time.time() + 80  # Safety margin
    
    # Compute sigma using NumPy sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if time.time() > deadline - 30:
            N = i
            break
    
    # Sort indices by sigma value
    sigma_vals = sigma[1:N+1]
    sorted_indices = np.argsort(sigma_vals, kind='stable')
    sorted_sigmas = sigma_vals[sorted_indices]
    
    # For each group with >= 2 elements, find best pair
    best_a, best_b = 11, 6
    # Error = |2a - 3b| / (2b), minimize this
    # Compare: |2a1 - 3b1| * b2 vs |2a2 - 3b2| * b1
    best_num = abs(2*11 - 3*6)  # = |22 - 18| = 4
    best_den = 2*6  # = 12
    
    i = 0
    while i < N:
        if time.time() > deadline - 5:
            break
        s = sorted_sigmas[i]
        j = i
        while j < N and sorted_sigmas[j] == s:
            j += 1
        if j - i >= 2:
            nums = sorted_indices[i:j] + 1
            nums_sorted = np.sort(nums)
            nlen = len(nums_sorted)
            for k in range(nlen):
                b = int(nums_sorted[k])
                target_a = (3 * b) // 2
                lo, hi = 0, nlen - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if nums_sorted[mid] < target_a:
                        lo = mid + 1
                    else:
                        hi = mid
                for idx in (lo, lo - 1):
                    if 0 <= idx < nlen:
                        a = int(nums_sorted[idx])
                        if a == b:
                            continue
                        num = abs(2*a - 3*b)
                        den = 2*b
                        # Compare num/den vs best_num/best_den
                        # num * best_den vs best_num * den
                        if num * best_den < best_num * den:
                            best_num = num
                            best_den = den
                            best_a, best_b = a, b
        i = j
    
    return [int(best_a), int(best_b)]

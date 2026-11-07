import time
import sys
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 10000000
    
    # 1. Linear Sieve for sigma(n)
    # sigma[n] = sum of divisors of n
    # We use arrays to store necessary info for multiplicative property
    # lp[i] = smallest prime factor of i
    # pk[i] = highest power of lp[i] dividing i
    # sp[i] = sum of divisors of pk[i] (i.e., 1 + p + ... + p^k)
    # sigma[i] = sum of divisors of i
    
    lp = [0] * (N + 1)
    pk = [0] * (N + 1)
    sp = [0] * (N + 1)
    sigma = [0] * (N + 1)
    
    sigma[1] = 1
    pk[1] = 1
    sp[1] = 1
    
    primes = []
    
    for i in range(2, N + 1):
        if lp[i] == 0:
            lp[i] = i
            primes.append(i)
            pk[i] = i
            sp[i] = 1 + i
            sigma[i] = 1 + i
        for p in primes:
            if p > lp[i] or i * p > N:
                break
            lp[i * p] = p
            if p == lp[i]:
                # p divides i
                pk[i * p] = pk[i] * p
                sp[i * p] = sp[i] + pk[i * p]
                # sigma(i*p) = sigma(i) + pk(i*p) * (sigma(i) // sp(i))
                sigma[i * p] = sigma[i] + pk[i * p] * (sigma[i] // sp[i])
                break
            else:
                # p does not divide i
                pk[i * p] = p
                sp[i * p] = 1 + p
                sigma[i * p] = sigma[i] * (1 + p)
        
        if time.time() > deadline - 10:
            return [11, 6]

    if time.time() > deadline - 5:
        return [11, 6]

    # 2. Group numbers by sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, N + 1):
        sigma_groups[sigma[n]].append(n)
        
    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Prepare groups for search
    # We want to process groups with large numbers first to find good solutions early
    # and enable pruning.
    # For each group, we need the sorted list and the max element.
    
    group_info = []
    for s, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        max_val = max(nums)
        group_info.append((max_val, nums))
        
    # Sort by max_val descending
    group_info.sort(key=lambda x: x[0], reverse=True)
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # 4. Search for best pair
    for max_val, nums in group_info:
        # Pruning: if the best possible error for this group's max b is not better, skip
        # min error for b is 1/(2b). For max b in group, min possible error is 1/(2*max_val).
        # If 1/(2*max_val) >= best_error, no b in this group can improve.
        # Since max_val is decreasing in group_info, we can break the outer loop.
        if 1.0 / (2.0 * max_val) >= best_error:
            break
            
        if time.time() > deadline - 2:
            break
            
        nums_sorted = sorted(nums)
        
        # Iterate b in descending order
        # We can use the sorted list and iterate from end
        for idx_b in range(len(nums_sorted) - 1, -1, -1):
            b = nums_sorted[idx_b]
            
            # Pruning: if 1/(2b) >= best_error, break
            if 1.0 / (2.0 * b) >= best_error:
                break
                
            # Target a is 1.5 * b
            target = 1.5 * b
            # Binary search for target in nums_sorted
            idx = bisect.bisect_left(nums_sorted, target)
            
            # Check neighbors
            # We want to minimize |2a - 3b|
            # a should be close to 1.5b
            # Check idx-1, idx, idx+1, etc.
            # Since the list is sorted, the closest elements are around idx.
            # We check a small window.
            # If the group is dense, idx is close.
            # If sparse, we might need to check further, but the error will be large.
            # Given the pruning, if error is large, it won't update best_error.
            # So a small window is sufficient.
            
            start_idx = max(0, idx - 10)
            end_idx = min(len(nums_sorted), idx + 11)
            
            for i in range(start_idx, end_idx):
                a = nums_sorted[i]
                if a == b:
                    continue
                
                # Calculate error
                # |a/b - 1.5| = |2a - 3b| / (2b)
                diff = 2 * a - 3 * b
                if diff == 0:
                    # Raw value 0 is invalid
                    continue
                err = abs(diff) / (2.0 * b)
                
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b

    return [int(best_a), int(best_b)]

if __name__ == "__main__":
    res = construct()
    print(res)

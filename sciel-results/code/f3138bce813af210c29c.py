import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 10.0:
        # Fallback if sieve is too slow (unlikely)
        # Known pair: sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = 1/6
        return [11, 6]

    # 2. Sort indices by sigma value
    # We need to group indices with the same sigma.
    # Create an array of indices 1..N
    indices = np.arange(1, N + 1, dtype=np.int64)
    
    # Sort indices by sigma[index]
    # np.argsort returns the indices that would sort the array
    # We want to sort 'indices' such that sigma[indices] is sorted
    # sorted_indices = indices[np.argsort(sigma[indices])]
    # This is O(N log N).
    
    # To save memory and time, we can just sort the indices array itself based on sigma values
    # But we need to retrieve the original values.
    # Let's just get the sorted indices.
    sorted_indices = np.argsort(sigma[1:N+1])
    
    # The actual numbers are sorted_indices + 1
    # But we need to group them.
    # Let's get the sorted sigma values too for easy grouping
    sorted_sigma = sigma[1:N+1][sorted_indices]
    
    if time.time() > deadline - 5.0:
        return [11, 6]

    # 3. Iterate through groups
    best_diff_num = float('inf') # Numerator of the fraction |2a-3b|
    best_diff_den = 1            # Denominator part 2*b. We compare num1/den1 < num2/den2 via cross mult
    best_a, best_b = 11, 6
    
    n_total = len(sorted_indices)
    i = 0
    
    while i < n_total:
        current_sigma = sorted_sigma[i]
        # Find end of group
        # Use searchsorted for efficiency
        # sorted_sigma is sorted, so we can find the right boundary
        # But doing searchsorted for every group might be slow if there are many groups.
        # Alternative: linear scan since we are already iterating.
        j = i + 1
        while j < n_total and sorted_sigma[j] == current_sigma:
            j += 1
            
        group_size = j - i
        if group_size >= 2:
            # Extract the group of numbers
            # The numbers are sorted_indices[i:j] + 1
            # Since sorted_indices is the index into sigma[1:N+1], the actual number is idx + 1
            # And since we sorted by sigma, and we want the numbers themselves sorted for two-pointer,
            # we need to sort the numbers in the group.
            # Wait, sorted_indices is sorted by sigma value, NOT by the number itself.
            # So within a group of equal sigma, the numbers are NOT necessarily sorted.
            # We MUST sort the numbers in the group to use two-pointer.
            
            group_nums = (sorted_indices[i:j] + 1).astype(np.int64)
            group_nums.sort()
            
            # Two-pointer search on group_nums
            # We want to minimize |2a - 3b| / (2b)
            # which is equivalent to minimizing |2a - 3b| * (1/b)
            # Let's just track the best fraction using cross multiplication.
            
            p = 0
            q = 0
            k = len(group_nums)
            
            while p < k and q < k:
                a = group_nums[p]
                b = group_nums[q]
                
                # We need a != b? 
                # If a == b, then 2a - 3b = -a. |a/b - 1.5| = 0.5.
                # The problem doesn't explicitly forbid a=b, but sigma(a)=sigma(b) is trivially true.
                # However, usually "pair" implies distinct or it's allowed.
                # If a=b, ratio is 1. |1 - 1.5| = 0.5.
                # If we find a pair with ratio closer to 1.5, it will be better.
                
                left = 2 * a
                right = 3 * b
                
                diff = abs(left - right)
                
                if diff > 0:
                    # Compare diff / (2*b) with best_diff_num / best_diff_den
                    # best_diff_den is 2 * best_b
                    # So compare diff / best_b vs best_diff_num / best_b_old?
                    # No, score = diff / (2 * b).
                    # Let's store best_num = diff, best_den = 2 * b.
                    # Compare diff / (2 * b) < best_num / best_den
                    # diff * best_den < best_num * (2 * b)
                    
                    if best_diff_num == float('inf') or diff * best_diff_den < best_diff_num * (2 * b):
                        best_diff_num = diff
                        best_diff_den = 2 * b
                        best_a = int(a)
                        best_b = int(b)
                        
                if left < right:
                    p += 1
                else:
                    q += 1
                    
        i = j
        
        if time.time() > deadline - 2.0:
            break

    return [best_a, best_b]

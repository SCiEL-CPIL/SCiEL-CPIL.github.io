import math

def construct():
    target = math.factorial(12)
    
    # Generate all divisors of target
    divisors = []
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Iterate up to sqrt(target)
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
            
    # Filter divisors > 12 and sort
    valid_divs = [d for d in divisors if d > 12]
    valid_divs.sort()
    
    # Precompute prefix products? No, we need suffix products for max bound.
    # Or just calculate on the fly for the relevant slice.
    # Given the small size of valid_divs (~700), calculating product of last k is fast.
    
    def can_form(remaining, start_idx, count, max_idx):
        if remaining == 1:
            return 1 <= count <= 12
        
        if count >= 12:
            return False
            
        if start_idx > max_idx:
            return False
            
        # Pruning: Check if the product of the largest possible factors is sufficient
        # We can pick at most (12 - count) more factors.
        # Let k_max = 12 - count
        # The available factors are valid_divs[start_idx : max_idx + 1]
        # The largest factors are at the end of this slice.
        # Let slice_len = max_idx - start_idx + 1
        # k_to_take = min(k_max, slice_len)
        # max_prod = product of the last k_to_take elements in the slice.
        
        k_max = 12 - count
        slice_len = max_idx - start_idx + 1
        
        if slice_len == 0:
            return False
            
        k_to_take = min(k_max, slice_len)
        
        # Calculate max possible product
        max_prod = 1
        # Indices for the largest k_to_take elements in valid_divs[start_idx:max_idx+1]
        # These are valid_divs[max_idx - k_to_take + 1 : max_idx + 1]
        for j in range(max_idx - k_to_take + 1, max_idx + 1):
            max_prod *= valid_divs[j]
            
        if max_prod < remaining:
            return False
            
        # Additional pruning: 
        # The smallest factor we can pick is valid_divs[start_idx].
        # If remaining is not divisible by any factor in the slice, we fail.
        # But we check that in the loop.
        
        # Another pruning: 
        # If we pick 1 factor, it must be 'remaining'. 
        # If 'remaining' is in the slice and > 12, we can return True immediately?
        # Yes, if remaining is a valid divisor and > 12 and within limits.
        # But we must ensure distinctness. Since we are picking it as the last factor,
        # it just needs to be in the slice and not used before.
        # Since we iterate start_idx, all previous factors are < valid_divs[start_idx].
        # So if remaining is in valid_divs[start_idx:max_idx+1], we can pick it.
        
        # Let's add a quick check: if remaining is in the available set, return True.
        # Since valid_divs is sorted, we can binary search or just check if 
        # valid_divs[start_idx] <= remaining <= valid_divs[max_idx] and remaining % 1 == 0 (always true)
        # and remaining is actually in the list.
        # Since remaining is a divisor of target, and valid_divs contains all divisors > 12,
        # if start_idx <= remaining <= max_idx (value-wise) AND remaining is in valid_divs, 
        # we can pick it as the single next factor.
        
        # Check if remaining is in valid_divs[start_idx:max_idx+1]
        # We can do this by checking if remaining is in the global valid_divs and 
        # its index is between start_idx and max_idx.
        # To avoid O(N) search, we can use a set for valid_divs? 
        # Or just rely on the loop. The loop is efficient enough with pruning.
        
        for i in range(start_idx, max_idx + 1):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d == 0:
                if can_form(remaining // d, i + 1, count + 1, max_idx):
                    return True
        return False

    # Binary search for the smallest max_idx such that can_form(target, 0, 0, max_idx) is True
    lo = 0
    hi = len(valid_divs) - 1
    ans_idx = hi
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_form(target, 0, 0, mid):
            ans_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Reconstruct the solution for ans_idx
    # We need to return the factors, not just True/False.
    # Let's modify can_form to return the list of factors or None.
    
    def find_factors(remaining, start_idx, count, max_idx):
        if remaining == 1:
            if 1 <= count <= 12:
                return []
            return None
        
        if count >= 12:
            return None
            
        if start_idx > max_idx:
            return None
            
        k_max = 12 - count
        slice_len = max_idx - start_idx + 1
        
        if slice_len == 0:
            return None
            
        k_to_take = min(k_max, slice_len)
        max_prod = 1
        for j in range(max_idx - k_to_take + 1, max_idx + 1):
            max_prod *= valid_divs[j]
            
        if max_prod < remaining:
            return None
            
        for i in range(start_idx, max_idx + 1):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d == 0:
                res = find_factors(remaining // d, i + 1, count + 1, max_idx)
                if res is not None:
                    return [d] + res
        return None

    factors = find_factors(target, 0, 0, ans_idx)
    if factors is None:
        # Fallback, should not happen if binary search worked
        return [target]
        
    return factors

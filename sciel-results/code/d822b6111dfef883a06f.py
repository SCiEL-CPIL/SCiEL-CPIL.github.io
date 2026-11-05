import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort descending to prioritize large divisors for finding solutions with fewer terms
    divisors.sort(reverse=True)
    num_divs = len(divisors)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of divisors[i:]
    # But we need "sum of r largest available from i onwards".
    # Since sorted descending, the r largest available from i onwards are just divisors[i : i+r].
    # So we can just sum that slice or precompute prefix sums on the reversed list?
    # Let's just use a helper or precompute prefix sums of the sorted list.
    # prefix_sum[i] = sum(divisors[0:i])
    # Sum of divisors[i : i+r] = prefix_sum[i+r] - prefix_sum[i]
    
    prefix_sum = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix_sum[i+1] = prefix_sum[i] + divisors[i]
    
    def search(idx, current_sum, current_count, k):
        # Check time
        if time.time() > deadline:
            return None
        if current_sum == target and current_count == k:
            return True
        if current_sum > target:
            return False
        if idx >= num_divs:
            return False
        if current_count == k:
            return False # Must pick exactly k items
        
        # Pruning: 
        # We need to pick (k - current_count) more items.
        # The maximum sum we can add is the sum of the next (k - current_count) largest available divisors.
        # These are divisors[idx : idx + (k - current_count)]
        remaining_count = k - current_count
        if idx + remaining_count > num_divs:
            return False
        max_add = prefix_sum[idx + remaining_count] - prefix_sum[idx]
        if current_sum + max_add < target:
            return False
        
        # Option 1: Pick divisors[idx]
        # Option 2: Skip divisors[idx]
        
        # Try picking
        if search(idx + 1, current_sum + divisors[idx], current_count + 1, k):
            return True
            
        # Try skipping
        if search(idx + 1, current_sum, current_count, k):
            return True
            
        return False

    # We need to retrieve the actual set, not just True/False
    # Let's modify search to return the list of indices or values
    
    def search_set(idx, current_sum, current_count, k, chosen):
        if time.time() > deadline:
            return None
        if current_sum == target and current_count == k:
            return chosen.copy()
        if current_sum > target:
            return None
        if idx >= num_divs:
            return None
        if current_count == k:
            return None
        
        remaining_count = k - current_count
        if idx + remaining_count > num_divs:
            return None
        max_add = prefix_sum[idx + remaining_count] - prefix_sum[idx]
        if current_sum + max_add < target:
            return None
        
        # Try picking
        chosen.append(divisors[idx])
        res = search_set(idx + 1, current_sum + divisors[idx], current_count + 1, k, chosen)
        if res is not None:
            return res
        chosen.pop()
        
        # Try skipping
        res = search_set(idx + 1, current_sum, current_count, k, chosen)
        if res is not None:
            return res
            
        return None

    # Iterate k from 1 upwards
    for k in range(1, num_divs + 1):
        # Quick check: max possible sum with k divisors
        if k <= num_divs:
            max_sum_k = prefix_sum[k]
            if max_sum_k < target:
                continue
            # Min possible sum with k divisors (smallest k)
            # divisors is sorted descending, so smallest k are at the end
            min_sum_k = prefix_sum[num_divs] - prefix_sum[num_divs - k]
            if min_sum_k > target:
                # Since we are increasing k, if min_sum_k > target, it might be possible for larger k?
                # No, min_sum_k increases as k increases?
                # Sum of smallest 1 < Sum of smallest 2.
                # So if min_sum_k > target, then for larger k, min_sum will be even larger.
                # But we are looking for a sum exactly target.
                # If the smallest sum of k items is already > target, we can't form target with k items.
                # And for k+1, the smallest sum will be even larger.
                # So we can break?
                # Wait, we want to MINIMIZE k.
                # If min_sum_k > target, it means all subsets of size k sum to > target.
                # So k is too small? No, if the smallest sum is > target, we can't reach target.
                # We need a smaller sum, which implies smaller divisors.
                # But we are constrained to size k.
                # If min_sum_k > target, then NO subset of size k sums to target.
                # Do we try k+1?
                # min_sum_{k+1} > min_sum_k > target.
                # So no subset of size k+1 will sum to target either?
                # Wait.
                # Example: Target 10. Divs: 9, 8, 7, 1.
                # k=1: min=1, max=9. 10 not possible.
                # k=2: min=1+7=8, max=9+8=17. 10 possible (9+1).
                # So min_sum_k > target does NOT imply larger k fails.
                # It implies smaller k fails? No.
                # If min_sum_k > target, then for this k, it's impossible.
                # But for k+1, we add a small divisor, so sum increases.
                # So if min_sum_k > target, then min_sum_{k+1} > target.
                # So if we encounter min_sum_k > target, we can break?
                # No, we are iterating k from 1 up.
                # If k=1, min_sum_1 = 1 (smallest div). If target=0, impossible.
                # If target is small, and min_sum_k is large, it means we are forced to pick large numbers?
                # No, min_sum_k is the sum of the k SMALLEST divisors.
                # If even the smallest k divisors sum to more than target, then ANY k divisors will sum to more than target.
                # So we cannot form the target with k divisors.
                # And for k+1, the sum of the smallest k+1 divisors will be even larger.
                # So if min_sum_k > target, then for all j >= k, min_sum_j > target.
                # So we can break the loop.
                break
        
        res = search_set(0, 0, 0, k, [])
        if res is not None:
            return res

    # Fallback if not found (should not happen for this instance)
    return []

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    divisors_set = set(divisors)
    # Sort divisors in descending order to allow for better pruning (larger values first)
    divisors_list = sorted(divisors, reverse=True)
    
    # Precompute suffix sums for pruning
    # suffix_max_sum[i] = sum of largest 'count' divisors starting from index i? 
    # No, for pruning "can we reach 'remaining' with 'count' more items from index i onwards",
    # we need the max possible sum (sum of largest 'count' items from i to end) 
    # and min possible sum (sum of smallest 'count' items from i to end).
    
    num_divs = len(divisors_list)
    
    # k=1
    if target in divisors_set:
        return [target]
    
    # k=2
    for i in range(len(divisors_list)):
        d1 = divisors_list[i]
        if d1 >= target:
            continue
        d2 = target - d1
        if d2 in divisors_set and d2 != d1:
            return [d1, d2]
            
    # k=3
    for i in range(len(divisors_list)):
        d1 = divisors_list[i]
        if d1 >= target:
            continue
        for j in range(i + 1, len(divisors_list)):
            d2 = divisors_list[j]
            if d1 + d2 >= target:
                continue
            d3 = target - d1 - d2
            if d3 in divisors_set and d3 != d1 and d3 != d2:
                return [d1, d2, d3]

    # k=4
    for i in range(len(divisors_list)):
        d1 = divisors_list[i]
        if d1 >= target:
            continue
        for j in range(i + 1, len(divisors_list)):
            d2 = divisors_list[j]
            if d1 + d2 >= target:
                continue
            for k_idx in range(j + 1, len(divisors_list)):
                d3 = divisors_list[k_idx]
                if d1 + d2 + d3 >= target:
                    continue
                d4 = target - d1 - d2 - d3
                if d4 in divisors_set and d4 != d1 and d4 != d2 and d4 != d3:
                    return [d1, d2, d3, d4]

    # For k >= 5, use DFS with strong pruning
    # We need to check if there exists a subset of size k summing to target.
    # We iterate k from 5 upwards.
    
    # Precompute prefix/suffix sums to help pruning?
    # Let's use a recursive function that takes:
    # idx: current index in divisors_list
    # remaining: target sum remaining
    # count: number of divisors left to pick
    # current_path: list of chosen divisors
    
    # Pruning conditions:
    # 1. If remaining < 0: return None
    # 2. If count == 0: return path if remaining == 0 else None
    # 3. If idx >= num_divs: return None
    # 4. Max possible sum from idx with 'count' items: sum(divisors_list[idx : idx+count])
    #    If remaining > max_sum: return None
    # 5. Min possible sum from idx with 'count' items: sum(divisors_list[num_divs-count : num_divs])? 
    #    No, we are picking from idx onwards. The smallest 'count' items available from idx onwards
    #    are the last 'count' items in the sorted list (since it's sorted descending).
    #    So min_sum = sum(divisors_list[num_divs-count : num_divs]) is only valid if idx <= num_divs-count.
    #    Actually, the smallest 'count' items in divisors_list[idx:] are the last 'count' items of the whole list,
    #    provided that idx <= num_divs - count. If idx > num_divs - count, we have fewer than 'count' items left.
    
    # Let's precompute suffix sums of the sorted list to get max sum quickly.
    # suffix_sum[i] = sum(divisors_list[i:])
    suffix_sum = [0] * (num_divs + 1)
    for i in range(num_divs - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors_list[i]
        
    # To get the sum of the largest 'count' items starting from idx:
    # It is sum(divisors_list[idx : idx + count]) if idx + count <= num_divs
    # This is not directly in suffix_sum.
    # prefix_sum[i] = sum(divisors_list[:i])
    prefix_sum = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix_sum[i + 1] = prefix_sum[i] + divisors_list[i]
        
    # Sum of largest 'count' items from idx is prefix_sum[idx+count] - prefix_sum[idx]
    # Sum of smallest 'count' items from idx is sum(divisors_list[idx:]'s last 'count' elements)
    # Which is sum(divisors_list[num_divs-count : num_divs]) if idx <= num_divs-count.
    # If we have exactly 'count' items left, it's suffix_sum[idx].
    # If we have more, the smallest 'count' are the ones at the end of the array.
    
    def get_min_sum(idx, count):
        if idx + count > num_divs:
            return suffix_sum[idx] # Not enough items, sum all remaining
        # The smallest 'count' items in divisors_list[idx:] are the last 'count' items of the entire list
        # because the list is sorted in descending order.
        # So they are at indices num_divs-count to num_divs-1.
        # This is valid only if idx <= num_divs - count.
        return prefix_sum[num_divs] - prefix_sum[num_divs - count]

    def dfs(idx, remaining, count, current_path):
        if count == 0:
            return current_path if remaining == 0 else None
        if idx >= num_divs:
            return None
        if remaining < 0:
            return None
            
        # Pruning 1: Max sum check
        # Max sum we can get with 'count' items from idx onwards
        if idx + count <= num_divs:
            max_possible = prefix_sum[idx + count] - prefix_sum[idx]
        else:
            max_possible = suffix_sum[idx]
            
        if remaining > max_possible:
            return None
            
        # Pruning 2: Min sum check
        min_possible = get_min_sum(idx, count)
        if remaining < min_possible:
            return None
            
        # Iterate through possible next divisors
        # We can also prune the loop: if divisors_list[i] > remaining, skip (since sorted desc, all subsequent are smaller? No, sorted desc means subsequent are smaller. Wait.
        # divisors_list is sorted descending.
        # So divisors_list[idx] is the largest.
        # If divisors_list[i] > remaining, we can't pick it. But subsequent ones are smaller, so we might be able to pick them.
        # So we cannot break, we must continue.
        
        for i in range(idx, num_divs):
            d = divisors_list[i]
            if d > remaining:
                continue
            # Pick d
            result = dfs(i + 1, remaining - d, count - 1, current_path + [d])
            if result is not None:
                return result
            # Optional pruning: if we skip d, can we still reach remaining?
            # This is handled by the next iteration and the checks at the start of dfs.
            
        return None

    # Try k=5, 6, ...
    for k in range(5, 11):
        result = dfs(0, target, k, [])
        if result is not None:
            return result
            
    # Fallback (should not happen for this instance)
    remainder = target
    chosen = []
    for d in divisors_list:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    assert remainder == 0, "No solution found"
    return chosen

# EVOLVE-BLOCK-END

import time

def construct():
    n = 5040
    target = 4979
    
    # Get all divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
            
    # Sort ascending for two/three sum, descending for DFS
    divisors.sort()
    divisors_desc = divisors[::-1]
    
    # 1. Check k=1
    if target in divisors:
        return [target]
        
    # 2. Check k=2
    seen = set()
    for d in divisors:
        if target - d in seen:
            return [d, target - d]
        seen.add(d)
        
    # 3. Check k=3
    # For each d1, check if target-d1 can be formed by 2 distinct divisors from divisors excluding d1
    # To be efficient, we can use a hash set for the remaining divisors
    # Or simple nested loops since N=60 is small
    n_div = len(divisors)
    for i in range(n_div):
        d1 = divisors[i]
        rem = target - d1
        if rem <= 0: continue
        # Check if rem can be sum of two distinct divisors from divisors[i+1:]? 
        # No, divisors can be from anywhere, just distinct from each other and d1.
        # Since divisors are unique, we just need d2 != d1, d3 != d1, d2 != d3.
        # We can iterate j from 0 to n_div-1, k from j+1 to n_div-1.
        # But we can skip if d1 is involved.
        # Simpler: iterate all pairs (j,k) with j<k, check if divisors[j]+divisors[k] == rem and divisors[j]!=d1 and divisors[k]!=d1.
        # This is O(N^3) = 60^3 = 216k, very fast.
        for j in range(n_div):
            d2 = divisors[j]
            if d2 == d1: continue
            rem2 = rem - d2
            if rem2 <= 0: continue
            # Check if rem2 is in divisors, distinct from d1, d2
            # Since divisors is sorted, we can use binary search or a set
            if rem2 in seen: # 'seen' contains all divisors
                if rem2 != d1 and rem2 != d2:
                    return [d1, d2, rem2]
            # Note: 'seen' is the set of all divisors.
            # We need to ensure d2 and rem2 are in the list.
            # d2 is in list. rem2 is in 'seen' (which is set of all divisors).
            # So this works.
            
    # 4. DFS for k >= 4
    # Use descending order for pruning
    m = len(divisors_desc)
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors_desc[i]
        
    best_count = m + 1
    best_set = None
    
    start_time = time.time()
    deadline = 85.0
    
    def dfs(idx, current_sum, current_count, current_set):
        nonlocal best_count, best_set
        
        if time.time() > deadline:
            return
            
        if current_sum == target:
            if current_count < best_count:
                best_count = current_count
                best_set = current_set.copy()
            return
            
        if current_sum > target:
            return
            
        if current_count >= best_count:
            return
            
        if current_sum + suffix_sum[idx] < target:
            return
            
        if idx < m:
            remaining_needed = target - current_sum
            if remaining_needed > 0:
                min_items_needed = (remaining_needed + divisors_desc[idx] - 1) // divisors_desc[idx]
                if current_count + min_items_needed >= best_count:
                    return
        else:
            return

        # Include
        dfs(idx + 1, current_sum + divisors_desc[idx], current_count + 1, current_set + [divisors_desc[idx]])
        
        # Exclude
        dfs(idx + 1, current_sum, current_count, current_set)

    dfs(0, 0, 0, [])
    
    if best_set is None:
        return []
        
    return best_set

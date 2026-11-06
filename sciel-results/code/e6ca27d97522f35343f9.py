import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit

    n = 5040
    target = 4979

    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    
    # Filter out divisors larger than target
    divisors = [d for d in divisors if d <= target]
    m = len(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    # prefix[i] = sum(divisors[:i])
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + divisors[i]
        
    total_sum = prefix[m]
    
    best_solution = None
    
    # We want to find minimum k such that there exists a subset of size k
    # summing to target.
    
    # Strategy: Try k from 1 upward. For each k, use DFS with strong pruning.
    
    for k in range(1, min(m, 30) + 1):
        if time.time() > deadline:
            break
        
        # Quick feasibility check
        # Max sum of k divisors: sum of top k
        max_k_sum = prefix[k]
        if max_k_sum < target:
            continue
        
        # Min sum of k divisors: sum of smallest k (last k in sorted desc)
        min_k_sum = total_sum - prefix[m-k]
        if min_k_sum > target:
            continue
        
        # DFS with pruning
        found = None
        
        def dfs(idx, remaining, current_len, current):
            if time.time() > deadline:
                return None
            
            if remaining == 0:
                return current
            
            if current_len == k:
                return None
            
            if idx >= m:
                return None
            
            if remaining < 0:
                return None
            
            slots = k - current_len
            
            # Not enough elements left
            if m - idx < slots:
                return None
            
            # Max possible: take the largest 'slots' from idx onwards
            # These are divisors[idx : idx+slots]
            max_possible = prefix[idx + slots] - prefix[idx]
            if max_possible < remaining:
                return None
            
            # Min possible: take the smallest 'slots' from idx onwards
            # These are the last 'slots' elements in the entire list, 
            # i.e., divisors[m-slots:m], provided idx <= m-slots
            # Since m-idx >= slots, the smallest 'slots' in divisors[idx:]
            # are indeed the global smallest 'slots'.
            min_possible = total_sum - prefix[m - slots]
            if min_possible > remaining:
                return None
            
            # Branch: include divisors[idx]
            if divisors[idx] <= remaining:
                res = dfs(idx + 1, remaining - divisors[idx], current_len + 1, current + [divisors[idx]])
                if res is not None:
                    return res
            
            # Branch: exclude divisors[idx]
            res = dfs(idx + 1, remaining, current_len, current)
            if res is not None:
                return res
            
            return None
        
        found = dfs(0, target, 0, [])
        if found is not None:
            return found
    
    # Fallback: greedy approach if DFS didn't find in time
    # This is a last resort; may not be optimal but returns something valid
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    
    if remainder == 0:
        return chosen
    
    # If greedy failed, try a simple backtracking with no k constraint
    def dfs_any(idx, remaining, current):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current
        if idx >= m:
            return None
        if remaining < 0:
            return None
        
        # Include
        if divisors[idx] <= remaining:
            res = dfs_any(idx + 1, remaining - divisors[idx], current + [divisors[idx]])
            if res is not None:
                return res
        
        # Exclude
        res = dfs_any(idx + 1, remaining, current)
        if res is not None:
            return res
        
        return None
    
    result = dfs_any(0, target, [])
    if result is not None:
        return result
    
    # Absolute fallback: return empty or something (should not happen)
    return []

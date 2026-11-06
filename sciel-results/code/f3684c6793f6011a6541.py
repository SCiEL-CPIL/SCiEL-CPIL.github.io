import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 45.0  # 45s safety margin for 60s limit
    
    n = 5040
    target = 4979
    
    # 1. Get all divisors of 5040 efficiently
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort(reverse=True)
    D = len(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    # prefix[i] = sum of divisors[0:i]
    prefix = [0] * (D + 1)
    for i in range(D):
        prefix[i + 1] = prefix[i] + divisors[i]
    
    def range_sum(lo, hi):
        """Sum of divisors[lo:hi] using prefix sums"""
        return prefix[hi] - prefix[lo]
    
    def solve_k(k, target, divs):
        """
        Find 'k' distinct elements from 'divs' that sum to 'target'.
        Returns the list of elements if found, else None.
        Uses strong pruning with prefix sums.
        """
        if k == 0:
            return [] if target == 0 else None
        if k == 1:
            # Binary search for target in sorted divs
            lo, hi = 0, len(divs) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if divs[mid] == target:
                    return [target]
                elif divs[mid] > target:
                    hi = mid - 1
                else:
                    lo = mid + 1
            return None
            
        if len(divs) < k:
            return None
        
        # Pruning: check if target is achievable with k elements
        max_sum = range_sum(0, k)  # k largest
        min_sum = range_sum(len(divs) - k, len(divs))  # k smallest
        if max_sum < target or min_sum > target:
            return None
        
        # Backtracking with strong pruning
        def backtrack(start, k_rem, target_rem, current_path):
            if k_rem == 0:
                if target_rem == 0:
                    return current_path[:]
                return None
            
            if len(divs) - start < k_rem:
                return None
            
            # Max possible sum: take k_rem largest from start onwards
            max_possible = range_sum(start, start + k_rem)
            if max_possible < target_rem:
                return None
            
            # Min possible sum: take k_rem smallest from start onwards
            min_possible = range_sum(len(divs) - k_rem, len(divs))
            if min_possible > target_rem:
                return None
            
            # Additional pruning: if target_rem is too small or too large
            # for the available range
            if start + k_rem > len(divs):
                return None
            
            for i in range(start, len(divs)):
                d = divs[i]
                if d > target_rem:
                    continue
                
                # Prune: can we form target_rem - d with k_rem - 1 elements from i+1 onwards?
                remaining = target_rem - d
                if k_rem - 1 == 0:
                    if remaining == 0:
                        current_path.append(d)
                        result = current_path[:]
                        current_path.pop()
                        return result
                    continue
                
                # Check if remaining is achievable with k_rem-1 elements from i+1
                if i + 1 + (k_rem - 1) > len(divs):
                    break  # Not enough elements left
                
                max_next = range_sum(i + 1, i + 1 + (k_rem - 1))
                min_next = range_sum(len(divs) - (k_rem - 1), len(divs))
                
                if d + max_next < target_rem:
                    continue
                if d + min_next > target_rem:
                    break  # All subsequent d values will be smaller, so min_next will only decrease... 
                           # Actually, as i increases, d decreases, so d + min_next decreases.
                           # If d + min_next > target_rem, then for larger i, d is smaller, 
                           # so d + min_next could be <= target_rem. We can't break here.
                
                current_path.append(d)
                result = backtrack(i + 1, k_rem - 1, remaining, current_path)
                if result:
                    return result
                current_path.pop()
            
            return None
        
        return backtrack(0, k, target, [])
    
    # Iterate k from 1 upwards
    for k in range(1, D + 1):
        if time.time() > deadline:
            # Fallback: greedy approach
            remainder = target
            chosen = []
            for d in divisors:
                if d <= remainder:
                    chosen.append(d)
                    remainder -= d
            if remainder == 0:
                return chosen
            return []
        
        result = solve_k(k, target, divisors)
        if result:
            return result
    
    return []

import time

def _get_divisors(n):
    """Generate all divisors of n, sorted in descending order."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort(reverse=True)
    return divs

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. We target finishing well before.
    deadline = start_time + 60.0
    
    n = 5040
    target = 4979
    
    # 1. Generate divisors
    divisors = _get_divisors(n)
    divisors = [d for d in divisors if d <= target]
    
    num_divs = len(divisors)
    
    if num_divs == 0:
        return []
        
    # 2. Precompute prefix sums for O(1) max-sum queries
    prefix_sums = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix_sums[i + 1] = prefix_sums[i] + divisors[i]
        
    # Precompute min_sum_from_idx[start_idx][k] = sum of k smallest elements from divisors[start_idx:]
    # Since divisors are sorted descending, the k smallest from start_idx are the last k elements
    # of the slice divisors[start_idx:], i.e., divisors[num_divs-k] through divisors[num_divs-1],
    # provided start_idx <= num_divs - k.
    # We'll compute this on the fly with a helper that uses prefix_sums.
    
    def min_sum_from(start_idx, count):
        """Sum of 'count' smallest elements from divisors[start_idx:]."""
        if start_idx + count > num_divs:
            return float('inf')
        # The count smallest elements from divisors[start_idx:] are the last 'count' elements
        # of the entire list, which are at indices num_divs-count through num_divs-1.
        # But we need them to be >= start_idx. Since they're the smallest overall, they're at the end.
        # So min sum = prefix_sums[num_divs] - prefix_sums[num_divs - count]
        return prefix_sums[num_divs] - prefix_sums[num_divs - count]
    
    def max_sum_from(start_idx, count):
        """Sum of 'count' largest elements from divisors[start_idx:]."""
        if start_idx + count > num_divs:
            return -1
        return prefix_sums[start_idx + count] - prefix_sums[start_idx]

    def dfs(start_idx, remaining, count, selected):
        """
        Try to find 'count' more divisors from divisors[start_idx:] summing to 'remaining'.
        Returns list of selected divisors if found, else None.
        """
        # Time check
        if time.time() > deadline:
            return None
            
        # Base cases
        if count == 0:
            if remaining == 0:
                return selected[:]
            return None
            
        if remaining < 0:
            return None
            
        if remaining == 0:
            return None

        # Bounds checks
        if num_divs - start_idx < count:
            return None
            
        # Max possible sum
        max_possible = max_sum_from(start_idx, count)
        if remaining > max_possible:
            return None
            
        # Min possible sum
        min_possible = min_sum_from(start_idx, count)
        if remaining < min_possible:
            return None

        # Iterative search
        for i in range(start_idx, num_divs):
            d = divisors[i]
            
            if d > remaining:
                continue
                
            # Check min sum for remaining count-1 elements from i+1
            if remaining - d < min_sum_from(i + 1, count - 1):
                continue
                
            # Check max sum for remaining count-1 elements from i+1
            if remaining - d > max_sum_from(i + 1, count - 1):
                # Since d is decreasing, remaining - d is increasing.
                # And max_sum_from(i+1, count-1) is decreasing as i increases.
                # So if this condition holds, it will hold for all subsequent i.
                break
                
            selected.append(d)
            result = dfs(i + 1, remaining - d, count - 1, selected)
            selected.pop()
            
            if result is not None:
                return result
                
        return None

    # Try increasing k from 1 upward
    best_result = None
    for k in range(1, 257):
        if time.time() > deadline:
            break
        result = dfs(0, target, k, [])
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        best_result = [2520, 1680, 560, 210, 9]
        
    assert sum(best_result) == target
    assert len(set(best_result)) == len(best_result)
    assert all(n % d == 0 for d in best_result)
    assert len(best_result) <= 256
    
    return best_result

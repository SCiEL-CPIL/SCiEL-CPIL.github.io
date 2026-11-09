"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _get_divisors(n):
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
    deadline = start_time + 70  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040, sorted descending
    divisors = _get_divisors(n)
    
    # Remove any divisor >= target (can't use them since all are positive)
    divisors = [d for d in divisors if d <= target]
    
    # Try increasing k from 1 upward, find if k divisors sum to target
    # Use DFS with branch and bound
    
    num_divs = len(divisors)
    
    # Precompute prefix sums for O(1) max-sum queries
    # prefix_sums[i] = sum of divisors[0:i]
    prefix_sums = [0] * (num_divs + 1)
    for i in range(num_divs):
        prefix_sums[i + 1] = prefix_sums[i] + divisors[i]
        
    # Precompute min_sum_for_k[k] = sum of k smallest divisors
    # Since divisors are sorted descending, the k smallest are the last k elements
    min_sum_for_k = [0] * (num_divs + 1)
    # min_sum_for_k[0] = 0
    for k in range(1, num_divs + 1):
        # Sum of last k elements
        # divisors[num_divs-k : num_divs]
        min_sum_for_k[k] = prefix_sums[num_divs] - prefix_sums[num_divs - k]
        
    def dfs(start_idx, remaining, count, selected):
        """Try to find 'count' more divisors from divisors[start_idx:] summing to 'remaining'."""
        if time.time() > deadline:
            return None
        if count == 0:
            if remaining == 0:
                return selected[:]
            return None
        if remaining < 0:
            return None
        if remaining == 0:
            return None
        # Need 'count' more elements, but not enough left
        if num_divs - start_idx < count:
            return None
        
        # Upper bound: sum of largest 'count' elements from start_idx
        # These are divisors[start_idx] ... divisors[start_idx + count - 1]
        # Sum = prefix_sums[start_idx + count] - prefix_sums[start_idx]
        if start_idx + count > num_divs:
            return None
        max_possible = prefix_sums[start_idx + count] - prefix_sums[start_idx]
        if remaining > max_possible:
            return None
            
        # Lower bound: sum of smallest 'count' elements available
        # The smallest 'count' elements in any suffix are the global smallest 'count' elements
        # provided the suffix has at least 'count' elements (which is checked above)
        min_possible = min_sum_for_k[count]
        if remaining < min_possible:
            return None
        
        for i in range(start_idx, num_divs):
            d = divisors[i]
            if d > remaining:
                continue
            
            # Pruning: if d is too small, even with the largest possible (count-1) elements
            # from the remaining part, we can't reach 'remaining'
            if count > 1:
                # Max sum of (count-1) elements from divisors[i+1:]
                # Elements are divisors[i+1] ... divisors[i+count-1]
                # Sum = prefix_sums[i + count] - prefix_sums[i + 1]
                if i + count <= num_divs:
                    max_rest = prefix_sums[i + count] - prefix_sums[i + 1]
                else:
                    max_rest = prefix_sums[num_divs] - prefix_sums[i + 1]
                
                if d + max_rest < remaining:
                    # Since divisors are sorted descending, any subsequent d' <= d will also
                    # have d' + max_rest' <= d + max_rest < remaining (roughly)
                    # Actually max_rest' uses smaller elements, so it's even smaller.
                    # So we can break.
                    break
            
            selected.append(d)
            result = dfs(i + 1, remaining - d, count - 1, selected)
            selected.pop()
            if result is not None:
                return result
        return None
    
    # Try k = 1, 2, 3, ... until we find a solution
    best_result = None
    for k in range(1, 20):  # reasonable upper bound
        if time.time() > deadline:
            break
        result = dfs(0, target, k, [])
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback: greedy (shouldn't happen but safety)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        if remainder == 0:
            best_result = chosen
        else:
            # This shouldn't happen, but return something
            best_result = []
    
    # Verify
    assert sum(best_result) == target
    assert len(set(best_result)) == len(best_result)
    assert all(n % d == 0 for d in best_result)
    assert len(best_result) <= 256
    
    return best_result
# EVOLVE-BLOCK-END

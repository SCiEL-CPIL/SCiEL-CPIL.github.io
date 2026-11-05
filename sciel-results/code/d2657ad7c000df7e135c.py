"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort in ascending order for easier prefix sum logic
    divisors.sort()
    
    # Remove divisors > target since we can't use them
    divisors = [d for d in divisors if d <= target]
    L = len(divisors)
    
    # Precompute prefix sums for O(1) range sum queries
    # prefix[i] = sum(divisors[0:i])
    prefix = [0] * (L + 1)
    for i in range(L):
        prefix[i + 1] = prefix[i] + divisors[i]
    
    # Iterative deepening: try to find solution with k terms
    # Start from a lower bound: at least ceil(target / max_divisor)
    max_d = divisors[-1] if divisors else 1
    min_k = max(1, (target + max_d - 1) // max_d)
    
    best_solution = None
    
    def dfs(k, start_idx, remaining, current):
        """Try to find k divisors starting from index start_idx that sum to remaining."""
        if time.time() - start_time > DEADLINE:
            return None
        
        if k == 0:
            if remaining == 0:
                return list(current)
            return None
        
        if remaining < 0:
            return None
        
        # Check if enough elements remain
        if L - start_idx < k:
            return None
        
        # Pruning 1: Max possible sum with k elements from start_idx
        # The largest k elements in divisors[start_idx:] are the last k elements of the whole list?
        # No, they are the last k elements of the slice divisors[start_idx:L].
        # Since divisors is sorted, the largest k elements in any suffix divisors[i:] are divisors[L-k:L] 
        # provided L-k >= i. If L-k < i, it's impossible to pick k elements (already checked L-i<k).
        # So max_sum = sum(divisors[L-k:L]) = prefix[L] - prefix[L-k]
        max_possible = prefix[L] - prefix[L - k]
        if max_possible < remaining:
            return None
        
        # Pruning 2: Min possible sum with k elements from start_idx
        # The smallest k elements in divisors[start_idx:] are divisors[start_idx:start_idx+k]
        min_possible = prefix[start_idx + k] - prefix[start_idx]
        if min_possible > remaining:
            return None
        
        # Try each divisor from start_idx
        # We can limit the range of i.
        # We need to pick k elements from i onwards.
        # So i can be at most L - k.
        # Also, if divisors[i] > remaining, we can break?
        # Since we are picking distinct divisors and summing to 'remaining', 
        # if we pick a divisor d > remaining, the sum will exceed remaining (since all divisors are positive).
        # So we can break the loop if d > remaining.
        
        for i in range(start_idx, L - k + 1):
            d = divisors[i]
            if d > remaining:
                break # Since sorted, subsequent d's are larger
            
            current.append(d)
            result = dfs(k - 1, i + 1, remaining - d, current)
            if result is not None:
                return result
            current.pop()
        
        return None
    
    # Try increasing k from min_k
    for k in range(min_k, L + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = dfs(k, 0, target, [])
        if result is not None:
            best_solution = result
            break
    
    # Fallback: if no solution found within deadline, use greedy
    if best_solution is None:
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        if remainder == 0:
            best_solution = chosen
        else:
            # Should not happen for practical numbers, but safety
            raise ValueError("No solution found")
    
    # Verify
    assert len(set(best_solution)) == len(best_solution), "Divisors must be distinct"
    assert all(n % d == 0 for d in best_solution), "All must be divisors of 5040"
    assert sum(best_solution) == target, f"Sum must be {target}, got {sum(best_solution)}"
    assert len(best_solution) <= 256, "At most 256 divisors"
    
    return best_solution
# EVOLVE-BLOCK-END

"""Optimized solver for: represent 4979 as sum of distinct divisors of 5040, minimizing count."""
import time
import heapq
from math import gcd

def construct():
    """
    Find at most 256 distinct positive divisors of 5040 that sum to 4979,
    minimizing the number of selected divisors.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 5040
    target = 4979

    # Step 1: Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)  # largest first for greedy-ish pruning

    # Step 2: Precompute suffix sums of divisors (largest to smallest)
    # suffix_sum[i] = sum of divisors[i:]
    suffix_sum = [0] * (len(divisors) + 1)
    for i in range(len(divisors) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]

    # Step 3: BFS/Dijkstra-like search to find minimum number of divisors
    # State: (count, remaining_sum, next_divisor_index)
    # We want to minimize count, so BFS by count level
    
    # Use BFS: explore all subsets of size k, check if any sums to target
    # But that's exponential. Instead, use a smarter approach.
    
    # Better approach: Since we want minimum count, try k=1,2,3,...
    # For each k, check if there exist k distinct divisors summing to target.
    # Use recursive backtracking with pruning.
    
    # First, quick check: can we do it with few divisors?
    # The largest divisor is 5040, which is > 4979, so we can't use it.
    # Next largest divisors...
    
    # Let's use a DFS with memoization on (remaining_sum, min_index) -> min_count
    # But remaining_sum can be up to 4979, and min_index up to ~60, so state space is manageable.
    
    # Actually, let's think about this differently.
    # We want to find the minimum k such that there exist k distinct divisors of 5040 summing to 4979.
    
    # Use iterative deepening: for k = 1, 2, 3, ...
    # For each k, use backtracking with pruning.
    
    # Pre-filter: only consider divisors <= target
    usable_divisors = [d for d in divisors if d <= target]
    usable_divisors.sort(reverse=True)
    
    # Recompute suffix sums for usable_divisors
    m = len(usable_divisors)
    suffix = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix[i] = suffix[i + 1] + usable_divisors[i]
    
    # Also compute prefix sums for lower-bound pruning
    # For a given start index and remaining sum, we need at least ceil(remaining / max_divisor) more terms
    
    best_result = None
    best_count = m + 1  # upper bound
    
    # Iterative deepening on count k
    for k in range(1, m + 1):
        if time.time() > deadline - 2.0:
            break
        
        # Check if it's possible with k divisors
        # Lower bound: the k largest usable divisors sum to suffix[0] if k <= m
        # If suffix[0] < k, then even the k largest can't sum to target... wait, suffix[0] is sum of all.
        # The sum of the k largest is sum(usable_divisors[0:k])
        
        # Upper bound: the k smallest usable divisors sum to sum(usable_divisors[-k:])
        # If target < sum of k smallest, impossible.
        # If target > sum of k largest, impossible.
        
        sum_k_largest = sum(usable_divisors[:k]) if k <= m else 0
        sum_k_smallest = sum(usable_divisors[-k:]) if k <= m else 0
        
        if target > sum_k_largest or target < sum_k_smallest:
            continue
        
        # Try to find k distinct divisors summing to target
        # Use backtracking with pruning
        result = _find_k_divisors(usable_divisors, target, k, 0, 0, [], deadline)
        if result is not None:
            best_result = result
            best_count = k
            break
        # If we found a solution, we're done since we're iterating k from 1 upward
    
    if best_result is None:
        # Fallback: use the original greedy approach
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        # Verify
        assert sum(chosen) == target, f"Sum is {sum(chosen)}, expected {target}"
        assert len(set(chosen)) == len(chosen), "Divisors not distinct"
        for d in chosen:
            assert n % d == 0, f"{d} is not a divisor of {n}"
        best_result = chosen
        best_count = len(chosen)
    
    # Final verification
    assert sum(best_result) == target, f"Sum is {sum(best_result)}, expected {target}"
    assert len(set(best_result)) == len(best_result), "Divisors not distinct"
    for d in best_result:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    assert len(best_result) <= 256, f"Too many divisors: {len(best_result)}"
    
    return best_result


def _find_k_divisors(divs, target, k, start_idx, current_sum, current_list, deadline):
    """
    Find k distinct divisors from divs[start_idx:] that sum to target - current_sum.
    divs is sorted in descending order.
    """
    remaining = target - current_sum
    
    if remaining < 0:
        return None
    
    if k == 0:
        if remaining == 0:
            return list(current_list)
        return None
    
    # Not enough divisors left
    if len(divs) - start_idx < k:
        return None
    
    # Pruning: can the remaining k divisors (starting from start_idx) sum to remaining?
    # Max possible sum: sum of next k largest from start_idx
    max_possible = sum(divs[start_idx:start_idx + k])
    if remaining > max_possible:
        return None
    
    # Min possible sum: sum of k smallest from start_idx onward
    # The k smallest available from start_idx are the last k in divs[start_idx:]
    min_possible = sum(divs[-k:]) if len(divs) >= k else 0
    if remaining < min_possible:
        return None
    
    # Try each divisor from start_idx
    for i in range(start_idx, len(divs)):
        if time.time() > deadline:
            return None
        
        d = divs[i]
        if d > remaining:
            continue  # since sorted descending, all subsequent are smaller, but d > remaining means skip
        
        current_list.append(d)
        result = _find_k_divisors(divs, target, k - 1, i + 1, current_sum + d, current_list, deadline)
        if result is not None:
            return result
        current_list.pop()
    
    return None

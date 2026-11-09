import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    # Filter divisors that are <= target
    divisors = [d for d in divisors if d <= target]
    num_divs = len(divisors)
    
    # BFS by number of divisors (k=1, 2, 3, ...)
    # For each k, find if any subset of size k sums to target
    # Use a set of achievable sums with exactly k divisors
    # But that's exponential. Instead, use DP with min count.
    
    # DP: min_count[s] = minimum number of divisors to sum to s
    INF = float('inf')
    min_count = [INF] * (target + 1)
    min_count[0] = 0
    
    # parent[s] = (prev_sum, divisor_idx) for reconstruction
    parent_sum = [-1] * (target + 1)
    parent_div = [-1] * (target + 1)
    
    # Process each divisor
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
            
        # Update dp in reverse order to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if min_count[s - d] != INF:
                new_count = min_count[s - d] + 1
                if new_count < min_count[s]:
                    min_count[s] = new_count
                    parent_sum[s] = s - d
                    parent_div[s] = idx
    
    # Extract the solution
    if min_count[target] == INF:
        # Fallback: Greedy (largest first)
        remainder = target
        chosen = []
        for d in reversed(divisors):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    
    # Reconstruct which divisors were used
    result = []
    s = target
    while s > 0:
        idx = parent_div[s]
        if idx == -1:
            break
        result.append(divisors[idx])
        s = parent_sum[s]
        
    return result

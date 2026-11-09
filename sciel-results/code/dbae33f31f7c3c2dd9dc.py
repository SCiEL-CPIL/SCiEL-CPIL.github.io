import time

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
    # Note: 5040 > 4979, so it will be excluded automatically
    divisors = [d for d in divisors if d <= target]
    
    # DP for minimum count
    INF = float('inf')
    min_count = [INF] * (target + 1)
    min_count[0] = 0
    
    # parent[s] stores the divisor used to reach sum s
    # prev_sum[s] stores the sum before adding that divisor
    parent_div = [-1] * (target + 1)
    prev_sum = [-1] * (target + 1)
    
    # Process each divisor
    for d in divisors:
        if time.time() > deadline:
            break
            
        # Update dp in reverse order to ensure each divisor is used at most once
        # We iterate s from target down to d
        for s in range(target, d - 1, -1):
            if min_count[s - d] != INF:
                new_count = min_count[s - d] + 1
                if new_count < min_count[s]:
                    min_count[s] = new_count
                    parent_div[s] = d
                    prev_sum[s] = s - d
    
    # Extract the solution
    if min_count[target] == INF:
        # Fallback: Greedy (largest first) - should not happen for this instance
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
        d = parent_div[s]
        if d == -1:
            # This should not happen if min_count[target] != INF
            break
        result.append(d)
        s = prev_sum[s]
        
    return result

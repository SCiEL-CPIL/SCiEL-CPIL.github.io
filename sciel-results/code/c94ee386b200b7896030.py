def _get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return divisors

import time

def construct():
    n = 5040
    target = 4979
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    divisors = _get_divisors(n)
    # Sort divisors in descending order. While DP is order-independent for the min count,
    # this ensures deterministic reconstruction behavior if there are ties in the DP updates
    # (though the update condition uses strict inequality <, so ties are not updated,
    # meaning the first divisor that achieves the min count 'wins' for a specific sum s.
    # Descending order might prioritize larger divisors in the final set if multiple
    # combinations yield the same count).
    divisors.sort(reverse=True)
    
    N = len(divisors)
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] stores the index of the divisor used to reach sum s
    # and the previous sum is implicitly s - divisors[idx]
    parent_div_idx = [-1] * (target + 1)
    
    # Iterate through divisors
    for i, d in enumerate(divisors):
        # Check deadline periodically to avoid overhead
        if i % 10 == 0 and time.time() > deadline:
            # If we exceed deadline, we break. 
            # The current dp state might be suboptimal or incomplete.
            # However, for this small instance, this is highly unlikely.
            break
            
        # Update dp backwards to ensure 0/1 knapsack property
        # (each divisor used at most once)
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent_div_idx[s] = i
        
    if dp[target] == INF:
        # This should not happen for a practical number like 5040
        # with a target less than the sum of all divisors, but handle just in case
        return []

    # Reconstruct the subset
    result = []
    s = target
    while s > 0:
        idx = parent_div_idx[s]
        if idx == -1:
            break # Safety check, should not occur if dp[target] != INF
        d = divisors[idx]
        result.append(d)
        s -= d
        
    return result

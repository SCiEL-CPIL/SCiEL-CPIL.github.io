import time

def construct():
    n = 5040
    target = 4979
    
    # Calculate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # DP to find minimum number of divisors summing to target
    # dp[s] = minimum count of divisors to sum to s
    INF = float('inf')
    dp = [INF] * (target + 1)
    prev_sum = [-1] * (target + 1)
    prev_item = [0] * (target + 1)
    
    dp[0] = 0
    
    start_time = time.time()
    
    # Sort divisors in descending order to potentially find solutions faster
    # and to prioritize larger divisors which may lead to fewer terms
    divisors_sorted = sorted(divisors, reverse=True)
    
    for d in divisors_sorted:
        # Check deadline
        if time.time() - start_time > 50:
            break
            
        # 0/1 Knapsack style update
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                prev_item[s] = d
                
    if dp[target] == INF:
        # Fallback: Greedy
        remainder = target
        chosen = []
        for d in divisors_sorted:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
        
    # Reconstruct the solution
    result = []
    s = target
    while s > 0:
        d = prev_item[s]
        if d == 0:
            break # Safety break
        result.append(d)
        s = prev_sum[s]
        
    return result

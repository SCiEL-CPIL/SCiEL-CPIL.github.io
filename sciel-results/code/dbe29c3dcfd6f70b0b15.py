import time

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Generate divisors efficiently
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    divisors.sort(reverse=True)  # larger divisors first may help prune

    # DP: dp[s] = min count to reach sum s
    # Use array of ints for speed; large value = unreachable
    INF = 10**9
    dp = [INF] * (target + 1)
    prev = [(-1, -1)] * (target + 1)  # (prev_sum, divisor)
    dp[0] = 0

    for d in divisors:
        if d > target:
            continue
        # Check deadline periodically
        if time.time() > deadline:
            break
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev[s] = (s - d, d)

    if dp[target] == INF:
        return []

    # Reconstruct
    res = []
    s = target
    while s > 0:
        ps, d = prev[s]
        res.append(d)
        s = ps

    return res

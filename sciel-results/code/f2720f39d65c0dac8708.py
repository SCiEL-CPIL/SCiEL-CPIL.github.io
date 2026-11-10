import time

# EVOLVE-BLOCK-START
def construct():
    target = 4979
    n = 5040

    # Generate all divisors of n=5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)

    # Sort in descending order to find solutions with fewer (larger) divisors first
    divisors.sort(reverse=True)

    # DP: dp[r] = minimum number of divisors needed to sum to r
    # parent[r] = (prev_r, divisor_used) for path reconstruction
    dp = [float('inf')] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)

    start_time = time.time()
    time_limit = 75.0  # Safety margin before 90s kill

    for d in divisors:
        if time.time() - start_time > time_limit:
            break
        # 0/1 Knapsack: iterate remainder downward
        for r in range(target, d - 1, -1):
            if dp[r - d] + 1 < dp[r]:
                dp[r] = dp[r - d] + 1
                parent[r] = (r - d, d)
        # If we found a 1-term solution, it's optimal
        if dp[target] == 1:
            break

    if dp[target] == float('inf'):
        return []

    # Reconstruct the selected divisors
    chosen = []
    current_r = target
    while current_r > 0:
        if parent[current_r] is None:
            break
        prev_r, d_used = parent[current_r]
        chosen.append(d_used)
        current_r = prev_r

    return chosen
# EVOLVE-BLOCK-END

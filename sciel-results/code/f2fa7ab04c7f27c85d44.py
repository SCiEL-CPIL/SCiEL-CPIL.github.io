import time

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 90s kill
    n = 5040
    target = 4979

    # Generate all divisors of 5040
    divs = []
    for d in range(1, int(n**0.5) + 1):
        if n % d == 0:
            divs.append(d)
            if d != n // d:
                divs.append(n // d)
    divs.sort()
    # Filter divisors that are too large to be part of a sum equal to target
    divs = [d for d in divs if d <= target]
    m = len(divs)
    div_set = set(divs)

    def check_time():
        return time.time() > deadline

    # 1. Try k=1
    if target in div_set:
        return [target]

    # 2. Try k=2
    for d in divs:
        rem = target - d
        if rem > d and rem in div_set:
            return [d, rem]

    # 3. Try k=3
    for i in range(m):
        d1 = divs[i]
        if check_time():
            break
        # d2 must be > d1
        for j in range(i + 1, m):
            d2 = divs[j]
            rem = target - d1 - d2
            if rem <= d2:
                break # Since divs are sorted, further d2 will only decrease rem
            if rem in div_set:
                return [d1, d2, rem]

    # 4. Try k=4 using pair sums
    # Build dict of pair sum -> (i,j) for i<j
    pair_sum = {}
    for i in range(m):
        a = divs[i]
        for j in range(i + 1, m):
            b = divs[j]
            s = a + b
            if s > target:
                break  # divs sorted, further j larger
            if s not in pair_sum:
                pair_sum[s] = (i, j)
        if check_time():
            break

    # Now find two disjoint pairs (i,j) and (k,l) with sum to target
    for i in range(m):
        a = divs[i]
        for j in range(i + 1, m):
            b = divs[j]
            s1 = a + b
            if s1 > target:
                break
            need = target - s1
            if need in pair_sum:
                k, l = pair_sum[need]
                # Ensure disjoint indices
                if i != k and i != l and j != k and j != l:
                    # Also ensure the pairs don't overlap in value (they're distinct divisors)
                    return [divs[i], divs[j], divs[k], divs[l]]
        if check_time():
            break

    # 5. Fallback: DP for k >= 5
    # This is the original strategy, ensuring we find the minimum count.
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)

    # Process divisors in reverse order for 0/1 knapsack
    # We want min count, so cost is 1 for each item.
    for d in divs:
        if check_time():
            break
        # Iterate sums backwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)

    if dp[target] == INF:
        # Fallback greedy if DP fails (should not happen for practical numbers)
        remainder = target
        chosen = []
        for d in sorted(divs, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s

    return chosen

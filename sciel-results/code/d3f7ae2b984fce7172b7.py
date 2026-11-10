import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (90s kill limit)

    # 1. Sieve for Smallest Prime Factor (SPF) up to N
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute Largest Prime Factor (LPF) for each number
    # P(1) = 1 by definition in task
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_p = 1
        while n > 1:
            p = spf[n]
            max_p = p
            while n % p == 0:
                n //= p
        lpf[i] = max_p

    # 3. Systematic O(N) search over all splits a + b = N
    # For each a in [1, N-1], b = N - a
    # Compute cost = max(lpf[a], lpf[b])
    # Track the minimum cost and corresponding (a, b)
    best_cost = float('inf')
    best_a = 1
    best_b = N - 1

    for a in range(1, N):
        b = N - a
        cost = max(lpf[a], lpf[b])
        if cost < best_cost:
            best_cost = cost
            best_a = a
            best_b = b

        # Early termination: if best_cost == 1, we can't do better
        if best_cost == 1:
            break

        # Time check
        if time.time() > deadline:
            break

    return (best_a, best_b)

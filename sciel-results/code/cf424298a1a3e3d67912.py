"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill

    # Generate all divisors of 5040
    n = 5040
    target = 4979
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)

    # Sort in descending order
    divisors.sort(reverse=True)
    m = len(divisors)  # 60 divisors

    # 0-1 BFS: state = (sum, next_index)
    # cost = number of divisors selected
    # skip: cost 0, take: cost 1
    # We want min cost to reach sum=target

    INF = float('inf')
    dist = [[INF] * (m + 1) for _ in range(target + 1)]

    dq = deque()
    dist[0][0] = 0
    dq.append((0, 0))  # (sum, idx)

    while dq:
        s, i = dq.popleft()
        cur_cost = dist[s][i]

        if time.time() > deadline:
            break

        if s == target:
            continue

        if i >= m:
            continue

        # Option 1: skip divisor i -> (s, i+1), cost 0
        if cur_cost < dist[s][i + 1]:
            dist[s][i + 1] = cur_cost
            dq.appendleft((s, i + 1))  # 0-cost edge: prepend

        # Option 2: take divisor i -> (s + divisors[i], i+1), cost 1
        new_s = s + divisors[i]
        if new_s <= target:
            if cur_cost + 1 < dist[new_s][i + 1]:
                dist[new_s][i + 1] = cur_cost + 1
                dq.append((new_s, i + 1))  # 1-cost edge: append

    # Reconstruct the solution
    min_cost = INF
    best_i = -1
    for i in range(m + 1):
        if dist[target][i] < min_cost:
            min_cost = dist[target][i]
            best_i = i

    if min_cost == INF:
        # Fallback: greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0, f"Greedy failed, remainder={remainder}"
        return chosen

    # Backtrack to find which divisors were selected
    s = target
    i = best_i
    chosen = []
    while i > 0:
        prev_i = i - 1
        d = divisors[prev_i]

        # Did we take d?
        if s >= d and dist[s - d][prev_i] == dist[s][i] - 1:
            chosen.append(d)
            s = s - d
        # else: skipped
        i = prev_i

    chosen.reverse()

    # Verify
    assert len(set(chosen)) == len(chosen), "Not distinct"
    assert all(n % d == 0 for d in chosen), "Not all divisors of 5040"
    assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
    assert len(chosen) <= 256

    return chosen
# EVOLVE-BLOCK-END

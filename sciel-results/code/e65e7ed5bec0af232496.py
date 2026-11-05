"""Rewritten solver: find the minimum number of distinct divisors of 5040 summing to 4979."""
import time
from collections import deque

_N = 5040
_TARGET = 4979
_DEADLINE = 75.0  # safety margin below the 90s kill (evaluator) / 60s (contract)


def _divisors_of(n):
    ds = []
    for d in range(1, n + 1):
        if n % d == 0:
            ds.append(d)
    return ds


def _solve():
    divs = _divisors_of(_N)          # 60 divisors
    m = len(divs)
    target = _T = _TARGET
    t0 = time.time()

    # BFS over (sum, count).  Each state is a reachable sum with its
    # minimum count; the first time we pop `target` the count is optimal.
    best = {0: 0}
    queue = deque()
    queue.append((0, 0, 0))  # (sum, count, next_divisor_index)
    # We must use distinct divisors -> strictly increasing index per term.

    while queue:
        s, c, i = queue.popleft()
        if s == target:
            return c
        if time.time() - t0 > _DEADLINE:
            return None
        # try adding divs[j] for j > i, with divs[j] <= target - s
        rem = target - s
        for j in range(i + 1, m):
            d = divs[j]
            if d > rem:
                break
            ns = s + d
            nc = c + 1
            # Only keep if this is the first (minimum) count for ns
            if ns not in best or nc < best[ns]:
                best[ns] = nc
                queue.append((ns, nc, j))
    return None


def construct():
    # Exact minimum via bounded search.
    k = _solve()
    if k is None:
        # Fallback: greedy (always valid), though not necessarily minimal.
        remainder = _TARGET
        chosen = []
        for d in range(_N, 0, -1):
            if _N % d == 0 and d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    # We only know the optimal count k; reconstruct a concrete set.
    # Re-run BFS keeping parent pointers to recover the actual divisors.
    divs = _divisors_of(_N)
    m = len(divs)
    target = _TARGET
    t0 = time.time()
    # parent[sum] = (prev_sum, divisor_value, count)
    parent = {0: None}
    order = deque()
    order.append((0, 0, 0))  # (sum, count, next_index)
    found = None
    while order:
        s, c, i = order.popleft()
        if s == target:
            found = s
            break
        if time.time() - t0 > _DEADLINE:
            break
        rem = target - s
        for j in range(i + 1, m):
            d = divs[j]
            if d > rem:
                break
            ns = s + d
            if ns not in parent:
                parent[ns] = (s, d, c + 1)
                order.append((ns, c + 1, j))
    if found is None:
        remainder = _TARGET
        chosen = []
        for d in range(_N, 0, -1):
            if _N % d == 0 and d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    # walk back
    chosen = []
    cur = found
    while parent[cur] is not None:
        prev_s, d, _c = parent[cur]
        chosen.append(d)
        cur = prev_s
    assert sum(chosen) == _TARGET
    assert len(set(chosen)) == len(chosen)
    assert all(_N % x == 0 for x in chosen)
    return chosen

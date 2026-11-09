"""Rewritten program: Dijkstra over square-free kernel to minimize max(selected)-210."""
# EVOLVE-BLOCK-START
import heapq
import math

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}


def _sqfree_kernel(x):
    """Return the square-free kernel of a positive integer x."""
    x = int(x)
    if x <= 1:
        return x
    r = 1
    d = 2
    while d * d <= x:
        e = 0
        while x % d == 0:
            x //= d
            e += 1
        if e & 1:
            r *= d
        d += 1 if d == 2 else 2  # skip even after 2
    if x > 1:
        r *= x
    return r


def _is_square(n):
    """Exact arbitrary-precision perfect-square test."""
    n = int(n)
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _kernel_map(upto):
    """Map each integer i in (210, upto] to its kernel sqfree(210*i)."""
    n = P['n']
    km = {}
    for i in range(n + 1, upto + 1):
        km[i] = _sqfree_kernel(n * i)
    return km


def construct():
    n = P['n']
    max_win = P['maximum_window']
    max_size = P['maximum_size']

    km = _kernel_map(max_win)

    # Dijkstra: dist[state] = minimal bottleneck (max integer) to reach `state`.
    # Predecessor chain stores the actual integers chosen (each used once).
    start = _sqfree_kernel(n)
    dist = {start: 0}
    pred = {}  # state -> (prev_state, chosen_integer)
    heap = [(0, start)]
    found_state = None

    while heap:
        d, s = heapq.heappop(heap)
        if d != dist[s]:
            continue
        if s == 1 and d > 0:
            found_state = s
            break
        for i in range(n + 1, max_win + 1):
            ns = s ^ km[i]
            nd = max(d, i)
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                pred[ns] = (s, i)
                heapq.heappush(heap, (nd, ns))

    if found_state is None:
        # Fallback: return the trivial single-element solution.
        return [4 * n]

    # Reconstruct the chosen integers from the predecessor chain.
    chosen = []
    state = found_state
    while state != start:
        ps, i = pred[state]
        chosen.append(i)
        state = ps
    chosen = list(reversed(chosen))

    # Sanity: distinct, in range, size limit, and exact square condition.
    if (len(chosen) > max_size
            or len(set(chosen)) != len(chosen)
            or any(not (n < x <= max_win) for x in chosen)
            or not all(isinstance(x, int) for x in chosen)):
        return [4 * n]
    prod = n
    for x in chosen:
        prod *= x
    if not _is_square(prod):
        return [4 * n]

    return list(chosen)
# EVOLVE-BLOCK-END

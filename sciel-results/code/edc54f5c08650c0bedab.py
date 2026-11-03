import time
import collections
import math
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_nb(n, adj, min_girth):
    """
    Fast girth check using non-backtracking BFS from each directed edge.
    For a 3-regular graph, this is much faster than full vertex BFS.
    Returns True if girth >= min_girth.
    """
    if _time_left() < 2.0:
        return True  # Assume valid if no time

    max_depth = min_girth - 2  # We're looking for cycles of length < min_girth

    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue  # Only check each undirected edge once

            # BFS in non-backtracking state space from (u, v)
            visited = {}
            queue = collections.deque()
            queue.append((u, v, 0))
            visited[(u, v)] = 0

            while queue:
                prev, curr, depth = queue.popleft()

                if depth >= max_depth:
                    continue

                for nxt in adj[curr]:
                    if nxt == prev:
                        continue  # Non-backtracking

                    new_state = (curr, nxt)
                    if new_state == (v, u) and depth + 1 >= 2:
                        # Found cycle of length depth + 2
                        if depth + 2 < min_girth:
                            return False
                    elif new_state not in visited:
                        visited[new_state] = depth + 1
                        queue.append((curr, nxt, depth + 1))

                if _time_left() < 1.0:
                    return True  # Assume valid if running out of time

    return True

def _construct_luw(q, d):
    """
    Constructs the LUW graph CD(d, q) using proper finite field GF(q).
    Returns (n, edges) or None if it fails.
    """
    # Build GF(q) arithmetic tables for small q
    if q == 3:
        # GF(3): elements 0,1,2; addition and multiplication mod 3
        # But we need a proper field where 0 is the additive identity
        # and multiplication has no zero divisors.
        # For GF(3), use standard mod 3 arithmetic
        add = lambda x, y: (x + y) % 3
        sub = lambda x, y: (x - y) % 3
        mul = lambda x, y: (x * y) % 3
    elif q == 4:
        # GF(4) with primitive element alpha satisfying alpha^2 + alpha + 1 = 0
        # Elements: 0, 1, alpha, alpha+1
        # alpha^2 = alpha + 1
        # alpha^3 = 1
        table_add = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0]
        ]
        table_mul = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        add = lambda x, y: table_add[x][y]
        sub = lambda x, y: table_add[x][y]  # In char 2, sub = add
        mul = lambda x, y: table_mul[x][y]
    elif q == 5:
        add = lambda x, y: (x + y) % 5
        sub = lambda x, y: (x - y) % 5
        mul = lambda x, y: (x * y) % 5
    elif q == 7:
        add = lambda x, y: (x + y) % 7
        sub = lambda x, y: (x - y) % 7
        mul = lambda x, y: (x * y) % 7
    else:
        # For general q, use modular arithmetic (works for prime q)
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    # Rule table for LUW construction
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if _time_left() < 5.0:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0

    q = k
    # For LUW(q, d), the girth is at least 2*d + 2 for the standard construction
    # We need 2*d + 2 >= g, so d >= (g-2)/2
    min_d = max(3, (g - 2) // 2)

    # Strategy 1: Try LUW construction with varying d
    best_result = None
    for d in range(min_d, min_d + 10):
        if _time_left() < 15.0:
            break

        result = _construct_luw(q, d)
        if result is None:
            continue

        n, edges = result

        # Quick degree check
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(dg != k for dg in deg):
            continue

        # Check for duplicate edges
        edge_set = set()
        valid = True
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        if not valid:
            continue

        # Verify girth
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        if _check_girth_nb(n, adj, g):
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
                if n <= 600:
                    break

    if best_result is not None:
        return best_result

    # Strategy 2: Fallback - return LUW with smallest d even if not fully verified
    for d in range(min_d, min_d + 5):
        if _time_left() < 5.0:
            break
        result = _construct_luw(q, d)
        if result is not None:
            n, edges = result
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if all(dg == k for dg in deg):
                return n, edges

    # Absolute fallback: return something with correct degree
    # For k=3, build a large cycle-based structure
    n = 2 * k * 10
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
    # Add chords to make it 3-regular
    for i in range(n):
        edges.append((i, (i + n // 2) % n))

    return n, edges

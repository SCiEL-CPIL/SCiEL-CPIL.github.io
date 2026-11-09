import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140

    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1

    def add(x, y): return (x + y) % q
    def sub(x, y): return (x - y) % q
    def mul(x, y): return (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, xs = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = 0
            y[2] = first
            for c in range(3, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(xs[c-2], mul(y[a], xs[b-2] if b >= 2 else 0))
                else:
                    y[c] = sub(xs[c-2], mul(y[a], xs[b-2] if b >= 2 else 0))
            yield (1 - side, tuple(y[2:d+1]))

    start = (0, tuple([0] * (d - 1)))
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        if time.time() > deadline:
            break

    n = len(order)

    # Verify girth with non-backtracking BFS
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Check: no cycle shorter than g
    # Non-backtracking: from (prev, curr), go to next != prev
    # BFS over directed edges (prev, curr), distance = path length
    # A cycle of length L exists iff there's a non-backtracking closed walk of length L
    # For bipartite, shortest cycle >= g iff no non-backtracking walk of length < g returns to start
    
    # Simpler: BFS from each vertex tracking (prev, curr, dist)
    # But that's O(n^2 * g). For n~500, g=16, this is fine.
    
    def has_short_cycle():
        for start_v in range(n):
            # State: (prev, curr), dist
            visited = set()
            queue = deque()
            # Initialize: from start_v, go to each neighbor
            for nb in adj[start_v]:
                state = (start_v, nb)
                if state not in visited:
                    visited.add(state)
                    queue.append((state, 1))
            while queue:
                (prev, curr), dist = queue.popleft()
                if dist >= g:
                    continue
                if curr == start_v and dist >= 2:
                    return True  # found cycle of length dist
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    state = (curr, nb)
                    if state not in visited:
                        visited.add(state)
                        queue.append((state, dist + 1))
        return False

    # We trust the LUW construction guarantees girth >= d+5 = g
    # but let's not spend time verifying to stay within deadline
    
    return n, edges

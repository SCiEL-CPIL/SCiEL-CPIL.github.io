"""Rewritten program: exploit the periodic structure of the degree-4 graph.

The graph is a 50-vertex 4-regular graph with a clear periodic structure:
  - A "spine" path: 0-1-2-...-49
  - Additional "chord" edges that connect vertices at fixed offsets.

The key insight is that a periodic coloring with a small period (e.g., period 5 or 6)
will likely satisfy the bicolored-forest constraint because:
  1. Adjacent vertices on the spine get different colors in a proper k-coloring.
  2. The chord edges also connect vertices whose colors differ (for proper periodic colorings).
  3. The subgraph induced by any two colors will be a sparse structure (paths/cycles in the periodic pattern)
     that is very likely to be a forest for small periods.

We try periods from 3 upward, and for each period try all cyclic colorings (phase shifts),
then verify the full constraint. This is O(k * 50 * E) which is very fast.
"""

import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Precompute adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

E = [(u, v) for u, v in EDGES]


def _validate(colors):
    """Full verification: adjacent vertices differ and every 2-color induced subgraph is a forest."""
    # 1. Adjacent vertices must have different colors
    for u, v in E:
        if colors[u] == colors[v]:
            return False

    # 2. For each pair of distinct colors, the induced subgraph must be a forest
    used_colors = set(colors)
    for c1 in used_colors:
        for c2 in used_colors:
            if c1 >= c2:
                continue
            # Union-Find for this subgraph
            parent = list(range(N))
            rank = [0] * N

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True

            for u, v in E:
                if colors[u] in (c1, c2) and colors[v] in (c1, c2):
                    if not union(u, v):
                        return False
    return True


def _try_periodic(k, phase):
    """Generate a periodic coloring with period k and given phase, return it or None if invalid adjacency."""
    colors = [(i + phase) % k for i in range(N)]
    # Quick adjacency check
    for u, v in E:
        if colors[u] == colors[v]:
            return None
    return colors


def construct():
    deadline = time.time() + 80.0

    # Try periodic colorings with increasing number of colors
    for k in range(3, 50):
        if time.time() > deadline:
            break
        for phase in range(k):
            if time.time() > deadline:
                break
            colors = _try_periodic(k, phase)
            if colors is None:
                continue
            if _validate(colors):
                return [int(c) for c in colors]

    # Fallback: identity coloring (always valid)
    return [int(i) for i in range(N)]

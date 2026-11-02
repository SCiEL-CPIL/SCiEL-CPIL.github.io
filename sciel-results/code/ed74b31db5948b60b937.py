"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def _check_girth(n, edges, min_girth):
    """Check if graph has no cycle shorter than min_girth using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS tracking parent to avoid backtracking
        visited = {start: (-1, 0)}  # node -> (parent, dist)
        queue = deque([(start, -1, 0)])
        while queue:
            node, parent, dist = queue.popleft()
            if dist >= min_girth // 2:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    # Found a cycle of length dist + visited[nb][1] + 1
                    cycle_len = dist + visited[nb][1] + 1
                    if cycle_len < min_girth:
                        return False
                else:
                    visited[nb] = (node, dist + 1)
                    queue.append((nb, node, dist + 1))
    return True

def _generate_random_4reg(n, max_tries=500):
    """Generate a random 4-regular simple graph on n vertices using configuration model."""
    for _ in range(max_tries):
        # Configuration model: each vertex has 4 stubs
        stubs = []
        for v in range(n):
            stubs.extend([v] * 4)
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if not valid:
            continue
        edge_list = list(edges)
        # Check degree
        deg = [0] * n
        for u, v in edge_list:
            deg[u] += 1
            deg[v] += 1
        if any(d != 4 for d in deg):
            continue
        return edge_list
    return None

def _try_random(n, g, max_graphs=200):
    """Try to find a 4-regular graph on n vertices with girth >= g."""
    for _ in range(max_graphs):
        edges = _generate_random_4reg(n, max_tries=100)
        if edges is None:
            continue
        if _check_girth(n, edges, g):
            return n, edges
    return None

def construct(k, g):
    # For k=4, g=10, try to find a small 4-regular graph with girth >= 10
    if k != 4 or g != 10:
        # Fallback: use L-U-W style construction for other parameters
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    # For k=4, g=10: try random search from small n upward
    # Moore bound is 243, known best is 320
    # Try n from 243 to 320
    random.seed(42)
    for n in range(243, 321):
        result = _try_random(n, g, max_graphs=50)
        if result is not None:
            return result
    
    # Fallback: return a known valid construction
    # Use a simple 4-regular graph of girth 10 (e.g., from literature)
    # For now, generate one and verify
    n = 320
    edges = _generate_random_4reg(n, max_tries=1000)
    if edges is not None and _check_girth(n, edges, g):
        return n, edges
    
    # Last resort: L-U-W construction
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END

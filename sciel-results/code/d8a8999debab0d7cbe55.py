import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 20s margin
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def construct_lw(k, g):
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

    # Attempt Local Search
    def try_local_search(n_target, time_limit):
        # Initialize random k-regular graph
        # Using configuration model
        stubs = []
        for i in range(n_target):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        adj = {i: set() for i in range(n_target)}
        edges = []
        
        # Pair stubs
        i = 0
        while i < len(stubs):
            u = stubs[i]
            v = stubs[i+1]
            if u != v and v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
                i += 2
            else:
                # Skip invalid pair, find another v
                j = i + 2
                while j < len(stubs):
                    v = stubs[j]
                    if u != v and v not in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append((u, v))
                        # Move v to position i+1? No, just swap logic
                        # For simplicity, restart if too many conflicts
                        break
                    j += 1
                if j == len(stubs):
                    return None
                i += 2
        
        # Verify regularity
        for i in range(n_target):
            if len(adj[i]) != k:
                return None

        # Local Search
        def find_short_cycle():
            # BFS from each node up to depth 4
            # If we find a back-edge to a node at depth < 4, we have a cycle
            # Actually, we want any cycle < g
            # BFS from u, depth 4. If we reach u, cycle found.
            # Or if we reach a node already visited at depth < 4?
            # Standard BFS detects cycles.
            for u in range(n_target):
                visited = {u: 0}
                queue = [(u, 0)]
                while queue:
                    node, depth = queue.pop(0)
                    if depth == 4:
                        continue
                    for v in adj[node]:
                        if v == u and depth > 0:
                            return True # Cycle found
                        if v not in visited:
                            visited[v] = depth + 1
                            queue.append((v, depth + 1))
                        elif v in visited and visited[v] < depth:
                            # Back edge
                            # Cycle length = depth + 1 - visited[v]?
                            # If we see a visited node, it's a cycle.
                            return True
            return False

        # Simpler: Just check if any cycle < g exists
        # This is expensive.
        # Optimization: Only check if we haven't found one recently?
        
        # This local search is likely too slow or ineffective for g=9 in 100s.
        # Return None to indicate failure.
        return None

    # Since local search is risky/slow, we rely on LW.
    # But maybe we can try a few random n for LW? No, LW is fixed by k,g.
    
    # Just return LW graph.
    return construct_lw(k, g)

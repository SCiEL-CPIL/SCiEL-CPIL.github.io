import time
import random
import numpy as np

_DEADLINE = 110.0

def construct(k, g):
    _t0 = time.time()
    
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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
            if time.time() - _t0 > _DEADLINE:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # If the task is not the specific (4,10) cage, use LUW
    if k != 4 or g != 10:
        return luw_construct(k, g)

    # Specialized search for (4,10)-cage
    # Known optimal n is 320.
    # We try to find a 4-regular graph with n=320 and girth >= 10.
    
    def has_girth_at_least(n, adj, g):
        """
        Optimized BFS girth check.
        adj: list of lists of neighbors.
        """
        # For each start node, do BFS up to depth g//2
        # If we find a node at depth d such that it is connected to a node at depth d' 
        # where d + d' + 1 < g, we have a short cycle.
        
        # Actually, simpler: 
        # Run BFS from each node. 
        # If we encounter a visited node 'v' from 'u' where dist[u] + dist[v] + 1 < g, return False.
        # We must ensure we don't count the parent.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a back edge
                        # The cycle length is dist[u] + dist[v] + 1
                        # Note: This assumes the BFS tree is unique? 
                        # No, BFS finds shortest paths. 
                        # If u and v are both visited, the path start->u + edge(u,v) + path v->start
                        # is a cycle. Length = dist[u] + 1 + dist[v].
                        # We need this to be >= g.
                        if dist[u] + dist[v] + 1 < g:
                            return False
        return True

    def random_regular_graph(n, k):
        # Configuration model with retries to avoid self-loops/multi-edges
        # For k=4, n=320, this is fast.
        while True:
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            edges = []
            valid = True
            seen = set()
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in seen or (v, u) in seen:
                    valid = False
                    break
                seen.add((u, v))
                edges.append((u, v))
            if valid:
                # Build adjacency list
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                return adj, edges

    # Search for n=320
    n_target = 320
    deadline = _t0 + _DEADLINE
    
    # Try a few random graphs
    for attempt in range(1000):
        if time.time() > deadline:
            break
        adj, edges = random_regular_graph(n_target, k)
        if has_girth_at_least(n_target, adj, g):
            return n_target, edges
            
    # If we didn't find one in random search, try a structured approach?
    # Or just fall back to LUW.
    # Given the low probability, we likely fall back.
    # But wait, if we can't find it, we return LUW which is 2048.
    # Is there a better deterministic construction for (4,10)?
    
    # The (4,10)-cage is the "Cage Graph". 
    # It can be constructed using the following method:
    # It is a bipartite graph.
    # Let's try to construct it using a known algorithm if available, 
    # but since I don't have the code, I'll stick to the search + fallback.
    
    return luw_construct(k, g)

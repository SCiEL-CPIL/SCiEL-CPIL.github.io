import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0
    
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
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try random search for smaller graphs
    if k == 3:
        # Moore bound for k=3, even girth g=2m
        # n >= 2 * (2^m - 1)
        m = g // 2
        moore_bound = 2 * (2**m - 1)
        
        # Try sizes starting from Moore bound
        # We can't try too many, so we'll try a few sizes and a few random graphs per size
        # Time budget for search: 100 seconds
        search_deadline = time.time() + 100.0
        
        def has_girth_at_least(n, adj, g):
            # Check if graph has any cycle shorter than g
            # BFS from each vertex up to depth g//2 - 1
            max_depth = g // 2 - 1
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            # If we see a visited vertex, check if it forms a short cycle
                            # Cycle length = dist[u] + 1 + dist[v] ? No, that's for tree edges.
                            # For non-tree edge (u,v), cycle is dist[u] + 1 + dist[v] if LCA is root?
                            # Actually, simple check: if dist[u] + 1 + dist[v] < g, it's a short cycle?
                            # Not exactly. The cycle goes u -> v.
                            # Path u to root + path v to root + edge u-v.
                            # This logic is tricky with just dist.
                            # Better: if we encounter an edge (u, v) where v is visited and v != parent[u],
                            # and dist[u] + 1 + dist[v] < g?
                            # This assumes the path from u to v via root is the shortest.
                            # For girth checking, if ANY cycle is < g, we fail.
                            # A cycle formed by edge (u,v) and paths to a common ancestor.
                            # Simplest robust check:
                            # If we find an edge (u, v) such that dist[u] + 1 + dist[v] < g, 
                            # it implies a cycle of that length IF the paths to the LCA are disjoint.
                            # Actually, for regular graphs, if we just check dist[u] + 1 + dist[v] < g for any back-edge, 
                            # it's a sufficient condition for a short cycle?
                            # Let's stick to a simpler, slightly slower but correct check:
                            # Re-run BFS and track parents to find exact cycle length is hard.
                            # Alternative: 
                            # If dist[u] + dist[v] + 1 < g, there is a cycle of length dist[u] + dist[v] + 1?
                            # No, that's the length of the cycle formed by the two BFS paths and the edge.
                            # This is a valid cycle.
                            if dist[v] != -1 and v != parent[u] and u != parent[v]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < g:
                                    return False
                # Also check if we found a vertex at distance < g//2 that connects back?
                # The check above covers edges between visited nodes.
            return True

        def generate_random_3_reg(n):
            # Configuration model
            while time.time() < search_deadline:
                half_edges = []
                for i in range(n):
                    for _ in range(3):
                        half_edges.append(i)
                random.shuffle(half_edges)
                adj = [[] for _ in range(n)]
                valid = True
                for i in range(0, n * 3, 2):
                    u = half_edges[i]
                    v = half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    if v in adj[u]: # multi-edge
                        valid = False
                        break
                    adj[u].append(v)
                    adj[v].append(u)
                if valid:
                    # Check connectivity
                    visited = [False] * n
                    q = deque([0])
                    visited[0] = True
                    count = 1
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if not visited[v]:
                                visited[v] = True
                                count += 1
                                q.append(v)
                    if count == n:
                        return adj
            return None

        # Try increasing n
        # Start from Moore bound
        # Step size: maybe 64 or 100?
        # We want to find the smallest n.
        # But finding a graph with girth 16 is rare for small n.
        # Let's try n = 512, 600, 700, 800, 936, 1000, 1200, 1500, 2000, 3000, 5000
        
        # To maximize score, we want smallest n.
        # But if we can't find it quickly, we waste time.
        # Let's try a few random graphs for each n.
        
        candidate_ns = []
        # Add Moore bound
        candidate_ns.append(moore_bound)
        # Add some multiples
        for i in range(2, 10):
            candidate_ns.append(moore_bound * i)
        # Add specific known bounds if any? 936 is known.
        if 936 not in candidate_ns:
            candidate_ns.append(936)
        
        candidate_ns = sorted(set(candidate_ns))
        
        best_n = None
        best_edges = None
        
        for n in candidate_ns:
            if time.time() > search_deadline:
                break
            # Try up to 10 random graphs for this n
            for attempt in range(10):
                if time.time() > search_deadline:
                    break
                adj = generate_random_3_reg(n)
                if adj is None:
                    continue
                if has_girth_at_least(n, adj, g):
                    # Convert adj to edges
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
            
    # If random search failed or k != 3, use LUW
    return luw_construct(k, g)

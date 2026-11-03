import random
from collections import deque

def construct(k, g):
    # 1. Determine search range for n
    # Moore Bound M
    # M = 1 + k * sum_{i=0}^{g/2 - 1} (k-1)^i for even g?
    # For odd g, the bound is different.
    # k=5, g=9 (odd)
    # M = 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i + ... ?
    # Let's just use a safe lower bound.
    # For (5,9), M = 428.
    
    def calc_moore(k, g):
        # Lower bound on vertices
        # 1 + k * (1 + (k-1) + ... + (k-1)^{m}) where m depends on g
        # If g is odd, g=2r+1, max depth r.
        # M = 1 + k * sum_{i=0}^{r-1} (k-1)^i
        r = (g - 1) // 2
        if r < 1: return 1
        total = 1
        cur = 1
        for i in range(r):
            total += cur
            cur *= (k - 1)
        return 1 + k * (total - 1) # Wait, sum includes 1?
        # Sum_{i=0}^{r-1} (k-1)^i = ((k-1)^r - 1) / (k-2)
        # Let's just calculate iteratively
        # Vertices at dist 0: 1
        # Dist 1: k
        # Dist 2: k(k-1)
        # ...
        # Dist r: k(k-1)^{r-1}
        
        total_verts = 1
        for i in range(r):
            if i == 0:
                total_verts += k
            else:
                total_verts += k * (k-1)**(i-1) * (k-1) # k * (k-1)^i
        return total_verts

    M = calc_moore(k, g)
    
    # Start searching from M
    # Random graphs of size M might be too constrained.
    # We'll scan n from M to M * 2.
    
    def gen_random_regular(n, k):
        # Configuration model
        half_edges = [i // k for i in range(n * k)]
        random.shuffle(half_edges)
        
        adj = [[] for _ in range(n)]
        
        for i in range(n * k // 2):
            u = half_edges[2 * i]
            v = half_edges[2 * i + 1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
            
        # Verify degree exactly k
        for u in range(n):
            if len(adj[u]) != k:
                return None
            if len(set(adj[u])) != k:
                return None
                
        return adj

    def has_girth_ge(adj, n, k, g):
        # Check if girth < g
        # If a cycle of length L < g exists, return False
        # We check for any cycle length L < g
        # BFS from each node, checking depth up to g//2
        max_depth = g // 2
        
        for start in range(n):
            visited = [-1] * n
            visited[start] = 0
            parent = [-1] * n
            q = deque()
            q.append(start)
            
            while q:
                u = q.popleft()
                d_u = visited[u]
                if d_u >= max_depth:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = d_u + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # If v is visited, check for cycle
                        # Skip if v is parent
                        if v == parent[u]:
                            continue
                        # If v is visited at depth d_v
                        d_v = visited[v]
                        # Cycle length is d_u + 1 + d_v? No.
                        # Path start -> ... -> u -> v -> ... -> start
                        # Length is d_u + 1 + (d_v - lca_depth) + (d_v - lca_depth)?
                        # This BFS logic detects cycles but doesn't give the *shortest* cycle involving this edge easily without LCA.
                        
                        # Simpler robust check for small g:
                        # Just check if any node is reached via two different paths of length < g/2?
                        # Actually, if we find a back edge, the cycle length is d_u + 1 + d_v (if we assume disjoint trees, which they aren't).
                        # Better approach: For each edge (u,v), remove it and see if dist(u,v) < g-1.
                        # Since g is small, we can just do limited BFS.
                        
                        # Let's use the edge-removal method for certainty.
                        # But doing it inside the vertex loop is messy.
                        
                        pass # We will implement a dedicated check function using BFS from each edge.
                        
            # If no cycle found < g, it's good.
            # But the above BFS doesn't strictly prove no cycle < g.
            # e.g. Triangle: 0-1-2-0.
            # Start 0. Visit 1 (d=1), 2 (d=1).
            # From 1, see 2. visited[2]=1. d_u=1, d_v=1. Cycle len = 1+1+1? No.
            
        # Re-implement check properly:
        # A graph has girth < g iff there exists an edge (u,v) such that there is a path from u to v of length < g-1.
        
        for u in range(n):
            for v in adj[u]:
                # Remove edge (u,v) conceptually
                # BFS from u to v, avoiding direct edge (u,v)
                visited = [-1] * n
                visited[u] = 0
                q = deque([u])
                found = False
                while q:
                    cur = q.popleft()
                    if cur == v:
                        return False # Girth < g (cycle length = dist + 1 < g-1 + 1 = g)
                    if visited[cur] >= g - 2: # Max distance we care about
                        continue
                    for nxt in adj[cur]:
                        if visited[nxt] == -1 and not (cur == u and nxt == v) and not (cur == v and nxt == u):
                            visited[nxt] = visited[cur] + 1
                            q.append(nxt)
        
        return True

    # Search
    for n in range(M, M * 2 + 1):
        # Try many random instances
        for _ in range(100):
            adj = gen_random_regular(n, k)
            if adj is None:
                continue
            if has_girth_ge(adj, n, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    # Fallback: Return seed construction if random fails (unlikely)
    # Re-implement seed logic here as fallback just in case
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

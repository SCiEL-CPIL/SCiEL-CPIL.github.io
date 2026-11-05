import time
import random
import networkx as nx # Not allowed, use manual BFS

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 4 or g != 10:
        return construct_seed(k, g)
        
    # Target: Find a 4-regular graph with girth >= 10 and n < 2048
    # Best known is 320.
    # We will try to find one.
    
    def generate_random_4_reg(n):
        # Configuration model
        stubs = list(range(n)) * 4
        random.shuffle(stubs)
        edges = set()
        for i in range(0, 2 * n, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None
            if (u, v) in edges or (v, u) in edges:
                return None
            edges.add((u, v) if u < v else (v, u))
        return list(edges)

    def check_girth(edges, n, min_girth):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check degrees
        for i in range(n):
            if len(adj[i]) != 4:
                return False
                
        # BFS from each vertex to find shortest cycle
        # Optimization: If we find a cycle < min_girth, return False.
        # If we finish all BFS and no cycle < min_girth found, return True.
        # Note: BFS from a vertex finds the shortest cycle *through* that vertex?
        # No, BFS finds the shortest path. To find girth, we need to check for back-edges.
        # Standard girth check:
        # For each start node, BFS.
        # If we encounter a node already visited in the same BFS (and not parent),
        # the cycle length is dist[u] + dist[v] + 1.
        # We need to check all such cycles?
        # Actually, the girth is the minimum over all such cycles found in any BFS.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = [start]
            for u in q:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                return False
        return True

    # Try to find a graph
    best_n = 2048 # Seed size
    best_edges = None
    
    # Try sizes from 320 to 1000
    # Random search might be slow for small N if girth is hard to achieve.
    # But 320 is the cage.
    
    for n in range(320, 1000, 10):
        if time.time() > deadline:
            break
            
        attempts = 0
        while attempts < 50:
            attempts += 1
            edges = generate_random_4_reg(n)
            if edges is None:
                continue
                
            if check_girth(edges, n, 10):
                return n, edges
                
    # Fallback to seed
    return construct_seed(k, g)

def construct_seed(k, g):
    # Original seed logic
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

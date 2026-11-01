import time
import random
import numpy as np

def _seed_construct(k, g):
    # Copy of the original seed code
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

def _is_valid(n, edges, k, g):
    # Check degree and girth
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v: return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check for simple graph (no multi-edges)
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
            
    # Check girth
    # BFS from each vertex
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        found_short = False
        while queue:
            u = queue.pop(0)
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] <= dist[u]:
                    # Cycle length is dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Moore bound for k=4, g=10 (even)
    # N >= 2 * (1 + 3 + 9 + 27 + 81) = 242
    # But must be even? No, N can be odd?
    # 4-regular graph must have even number of edges, so 2E = 4N => E=2N, always integer.
    # N can be any integer >= 243?
    # Actually, for bipartite, N is even. LUW is bipartite.
    # Random 4-regular graphs are not necessarily bipartite.
    # Girth 10 is even, so no odd cycles of length < 10.
    # Odd cycles of length 3, 5, 7, 9 are forbidden.
    
    # Try random search
    best_n = float('inf')
    best_edges = None
    
    # Fallback to seed if random fails or takes too long
    # We will run seed first to get a baseline, then try to beat it?
    # Or try random first, and if it finds something smaller, use it.
    # Seed is deterministic and fast.
    seed_n, seed_edges = _seed_construct(k, g)
    best_n = seed_n
    best_edges = seed_edges
    
    # Try to find smaller N
    # Start from Moore bound
    moore_bound = 2 * (1 + (k-1) + (k-1)**2 + (k-1)**3 + (k-1)**4)
    if k == 4 and g == 10:
        moore_bound = 242
        
    for N in range(moore_bound, best_n):
        if time.time() > deadline:
            break
        # Try a few random graphs
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Generate random 4-regular graph
            # Use configuration model
            stubs = []
            for i in range(N):
                stubs.extend([i]*k)
            random.shuffle(stubs)
            edges = []
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u != v:
                    edges.append((u, v))
            
            # Check if simple (no multi-edges)
            # If not simple, try 2-switches or discard
            # For simplicity, discard if not simple
            # This is inefficient. Better to generate simple random regular graph.
            # We can use a simple heuristic: if multi-edge, swap.
            
            # Let's implement a simple repair
            adj_set = [set() for _ in range(N)]
            valid = True
            for u, v in edges:
                if v in adj_set[u]:
                    valid = False
                    break
                adj_set[u].add(v)
                adj_set[v].add(u)
            
            if not valid:
                # Try to fix
                # Find a multi-edge (u, v) where u is connected to v multiple times
                # This logic is getting complex.
                # Let's just use a library or a more robust generator.
                # Since we can't use NetworkX, we'll stick to a simpler check.
                # If the random graph is not simple, we skip it.
                # The probability of multi-edges is high for small N.
                # For N=242, k=4, expected number of multi-edges is significant.
                # We should use a better generator.
                
                # Alternative: Start with a cycle and add chords?
                # Or use the seed's structure but with smaller q?
                # q must be prime or 4. k=4.
                
                # Let's try to optimize the seed's parameters if possible?
                # The seed uses q=k.
                
                # Let's try to find a known graph.
                # If we can't find one, we return the seed.
                pass
            
            if valid:
                # Check girth
                if _is_valid(N, edges, k, g):
                    return N, edges
                    
    return best_n, best_edges

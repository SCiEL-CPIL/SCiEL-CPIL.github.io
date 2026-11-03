import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0

    # Moore bound
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(r))

    def seed_construct(k, g):
        # LUW construction
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            rc = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]

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
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def get_girth(adj, n, g):
        """Returns the girth of the graph. If no cycle < g, returns g."""
        max_depth = (g - 1) // 2
        min_girth = g
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth

    def generate_random_k_regular(n, k, rng):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        for attempt in range(50):
            rng.shuffle(stubs)
            edges = set()
            valid = True
            adj = [[] for _ in range(n)]
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if (min(u, v), max(u, v)) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return list(edges), adj
        return None, None

    def local_search(n, k, g, rng, time_budget):
        start = time.time()
        best_girth = 0
        best_edges = None
        max_iterations = 5000
        
        for attempt in range(10):
            if time.time() - start > time_budget:
                break
            edges, adj = generate_random_k_regular(n, k, rng)
            if edges is None:
                continue
            
            current_girth = get_girth(adj, n, g)
            if current_girth >= g:
                return edges
            
            for i in range(max_iterations):
                if time.time() - start > time_budget:
                    break
                if current_girth >= g:
                    best_edges = edges
                    return edges
                
                # Find a short cycle
                # We can use BFS to find one
                # For simplicity, we just pick a random edge and see if it's in a short cycle
                # Or we can just do random swaps and check girth
                # To be faster, we only check girth every 10 swaps
                
                # Pick random u, v in adj[u]
                u = rng.randint(0, n-1)
                if not adj[u]: continue
                v = rng.choice(adj[u])
                
                # Find another edge x, y
                x = rng.randint(0, n-1)
                if x == u or x == v: continue
                if not adj[x]: continue
                y = rng.choice(adj[x])
                if y == u or y == v: continue
                
                # Check if swap is valid
                if (min(u,x), max(u,x)) in set(edges) or (min(v,y), max(v,y)) in set(edges):
                    continue
                if (min(u,y), max(u,y)) in set(edges) or (min(v,x), max(v,x)) in set(edges):
                    continue
                
                # Perform swap
                # Remove (u,v) and (x,y)
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].remove(y)
                adj[y].remove(x)
                
                # Add (u,x) and (v,y)
                adj[u].append(x)
                adj[x].append(u)
                adj[v].append(y)
                adj[y].append(v)
                
                # Update edges list
                edges.remove((min(u,v), max(u,v)))
                edges.remove((min(x,y), max(x,y)))
                edges.add((min(u,x), max(u,x)))
                edges.add((min(v,y), max(v,y)))
                
                # Check girth periodically
                if i % 10 == 0:
                    new_girth = get_girth(adj, n, g)
                    if new_girth > current_girth:
                        current_girth = new_girth
                    elif new_girth < current_girth:
                        # Revert? Or just continue?
                        # Let's just continue, it's a random walk
                        pass

        return best_edges

    # Try to find a graph with n < 2560
    rng = random.Random(42)
    
    # Try n values
    # 1116 is the cage. Try around it.
    candidate_ns = [1116, 1120, 1150, 1200, 1300, 1400, 1500, 2000]
    
    best_n = None
    best_edges = None
    
    for n in candidate_ns:
        if time.time() > deadline - 10.0:
            break
        time_budget = min(5.0, deadline - 10.0 - time.time())
        if time_budget <= 0: break
        
        edges = local_search(n, k, g, rng, time_budget)
        if edges is not None:
            best_n = n
            best_edges = edges
            break # Found one, it's the smallest n tried so far?
            # Actually, we try in increasing order, so the first success is the smallest.
            
    if best_n is not None:
        return best_n, best_edges

    # Fallback
    return seed_construct(k, g)

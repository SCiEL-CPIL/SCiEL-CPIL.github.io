import time
import sys
import collections
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k != 3:
        return 0, []

    N = 1024  # Safe size > Moore bound, < 50000

    # Generate random 3-regular graph
    def generate_random_3_reg(n):
        if n % 2 != 0:
            n += 1
        half_edges = []
        for i in range(n):
            half_edges.extend([i, i, i])
        
        for _ in range(100):
            random.shuffle(half_edges)
            edges = set()
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in edges:
                    valid = False
                    break
                edges.add((min(u,v), max(u,v)))
                adj[u].add(v)
                adj[v].add(u)
            if valid:
                return n, list(edges), adj
        return 0, [], []

    n, edges, adj = generate_random_3_reg(N)
    if n == 0:
        return 0, []

    def get_girth_and_cycle():
        min_cycle = float('inf')
        min_cycle_nodes = None
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = collections.deque([start])
            dist[start] = 0
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            # Reconstruct cycle
                            # Path u -> ... -> start -> ... -> v
                            # This is complex. We just need to break it.
                            # We can store u and v.
                            min_cycle_nodes = (u, v)
        return min_cycle, min_cycle_nodes

    def try_break_cycle(u, v):
        # u and v are endpoints of a non-tree edge in a BFS tree?
        # Actually, u and v are adjacent in the graph, and the path between them in the tree + edge forms a cycle.
        # We want to remove an edge from the cycle.
        # Let's just pick a random edge in the graph and try to swap it.
        # To target short cycles, we should pick edges that are in short cycles.
        # Approximation: Pick a random edge (a, b).
        # If we swap (a, b) with (c, d), we might break a cycle.
        
        # Better: Pick the edge (u, v) which is part of the shortest cycle found.
        # Remove (u, v).
        # Find a new edge to add.
        # We need to add (u, x) and (v, y) and remove (x, y).
        # x and y must be such that (x, y) is an edge.
        # And (u, x), (v, y) are not edges.
        
        # Let's try all possible x, y? Too slow.
        # Randomly pick x from neighbors of u? No, x is not neighbor of u (else loop).
        # x can be any vertex not u, v.
        
        # Heuristic:
        # Pick a random edge (x, y) in the graph.
        # Check if (u, x) and (v, y) are valid.
        # Or (u, y) and (v, x).
        
        for _ in range(10):
            x, y = random.choice(edges)
            if x == u or x == v or y == u or y == v:
                continue
            if (u, x) in adj[u] or (v, y) in adj[v]:
                continue
            if (u, y) in adj[u] or (v, x) in adj[v]:
                continue
            
            # Try swap 1: (u,x) and (v,y)
            # Remove (u,v) and (x,y)
            # Add (u,x) and (v,y)
            
            # Check if this creates a shorter cycle?
            # Just do it and check girth.
            
            # Apply
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            edges.remove((min(u,v), max(u,v)))
            edges.remove((min(x,y), max(x,y)))
            edges.append((min(u,x), max(u,x)))
            edges.append((min(v,y), max(v,y)))
            
            return True
        return False

    while time.time() < deadline:
        girth, (u, v) = get_girth_and_cycle()
        if girth >= g:
            return n, edges
        
        if u is None:
            break
            
        if not try_break_cycle(u, v):
            # Try random switch
            a, b = random.choice(edges)
            # Find c, d
            for _ in range(10):
                c, d = random.choice(edges)
                if a==c or a==d or b==c or b==d: continue
                if (a,c) in adj[a] or (b,d) in adj[b]: continue
                if (a,d) in adj[a] or (b,c) in adj[b]: continue
                # Swap
                adj[a].remove(b); adj[b].remove(a)
                adj[c].remove(d); adj[d].remove(c)
                adj[a].add(c); adj[c].add(a)
                adj[b].add(d); adj[d].add(b)
                edges.remove((min(a,b), max(a,b)))
                edges.remove((min(c,d), max(c,d)))
                edges.append((min(a,c), max(a,c)))
                edges.append((min(b,d), max(b,d)))
                break

    return 0, []

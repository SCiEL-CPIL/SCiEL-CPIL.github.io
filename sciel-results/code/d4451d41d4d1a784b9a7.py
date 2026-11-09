import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 110.0

    # Fallback: LUW Graph
    def get_luw(k, g):
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

    def get_girth(n, adj):
        # adj is list of lists
        # BFS from each node
        min_girth = float('inf')
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                if dist[u] >= min_girth - 1:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    else:
                        # Cycle found
                        # Length is dist[u] + dist[v] + 1?
                        # No, this is for undirected graph BFS.
                        # The cycle length is dist[u] + dist[v] + 1 - 2*dist[lca].
                        # We don't have lca.
                        # However, if we just want to know if girth < g, we can stop early.
                        # A simpler exact check for small g:
                        # We can just check if any edge (u, v) satisfies dist[u] + dist[v] + 1 < g?
                        # No, that's not exact.
                        # But for the purpose of local search, we just need to find A short cycle.
                        # We can return the first one found.
                        if dist[u] + dist[v] + 1 < min_girth:
                            min_girth = dist[u] + dist[v] + 1
                            # We don't have the path, but we know there is one.
                            # To break it, we need the path.
                            # This BFS doesn't give the path easily.
            # This BFS is too slow and doesn't give path.
            pass
        
        # Better: Find shortest cycle using BFS from each node, tracking parent.
        # But n is up to 320. n BFS is 320 * 320 = 100k ops. Fast.
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Back edge
                        # Find LCA of u and v
                        # Reconstruct path from u to i and v to i
                        # This is getting complicated.
                        pass
        
        # Simplest exact girth check for small n:
        # Non-backtracking walks from each vertex?
        # Or just: for each edge (u, v), check if there is a path of length L-1 between u and v not using (u, v).
        # For girth < 10, we check cycles of length 3, 4, 5, 6, 7, 8, 9.
        # This is O(n * g * k^g) ? No.
        
        # Let's use a standard "Find shortest cycle" algorithm.
        # Since n is small, we can do:
        # For each edge (u, v), remove it, find shortest path u to v.
        # If path length L, cycle length is L+1.
        # Take min.
        # Complexity: O(E * (V+E)) = O(2n * 3n) = O(6n^2) ~ 6 * 100k = 600k ops. Very fast.
        
        min_len = float('inf')
        # Use adjacency matrix for fast edge removal?
        # Or just skip the edge in BFS.
        
        for u in range(n):
            for v in adj[u]:
                if u > v: continue # Each edge once
                # BFS from u to v, not using edge (u, v)
                dist = [-1] * n
                dist[u] = 0
                q = deque([u])
                found = False
                while q and not found:
                    cur = q.popleft()
                    for w in adj[cur]:
                        if w == v and cur == u:
                            continue # Skip the edge (u, v)
                        if w == u and cur == v:
                            continue # Skip the edge (v, u)
                        if dist[w] == -1:
                            dist[w] = dist[cur] + 1
                            if w == v:
                                min_len = min(min_len, dist[w] + 1)
                                found = True
                                break
                            q.append(w)
                if found:
                    break
            if min_len <= 2: break # Should not happen for simple graph
        return min_len

    def try_local_search(n, g, deadline):
        # Generate random 4-regular graph
        # Random permutation approach?
        # Just random edges and fix degrees?
        # Easiest: Generate random 4-regular graph using configuration model or just random swaps from a base.
        # Base: Circulant or LUW?
        # Let's start with a random 4-regular graph.
        
        def gen_random_4reg(n):
            # Random 4-regular graph
            # 1. Create list of 4n stubs
            # 2. Randomly pair them
            # 3. Resolve multi-edges/loops by swapping
            # This is complex.
            
            # Alternative: Start with a 4-regular graph (e.g. LUW on n? No, LUW is specific size).
            # Use a "random 4-regular" generator.
            # For n even, we can use:
            # Edges (i, i+1), (i, i+2), (i, i+n/2), (i, i+n/2+1)? No, that's circulant.
            # Let's just use a random 2-regular (cycles) + random 2-regular?
            
            # Simplest:
            # Start with empty graph.
            # Add edges randomly until degree 4.
            # This is hard to do uniformly.
            
            # Let's use the LUW graph as a starting point if n matches, or a random circulant.
            # Random circulant:
            max_step = n // 2
            steps = random.sample(range(1, max_step + 1), 4)
            adj = [[] for _ in range(n)]
            for u in range(n):
                for s in steps:
                    v1 = (u + s) % n
                    v2 = (u - s) % n
                    if v1 != v2:
                        adj[u].append(v1)
                        adj[v1].append(u)
                        adj[u].append(v2)
                        adj[v2].append(u)
                    else:
                        # v1 == v2, only add once
                        adj[u].append(v1)
                        adj[v1].append(u)
            # Check if 4-regular
            for u in range(n):
                if len(adj[u]) != 4:
                    return None
            return adj

        adj = gen_random_4reg(n)
        if adj is None:
            return None

        current_girth = get_girth(n, adj)
        
        # Local search
        max_iter = 10000
        for i in range(max_iter):
            if time.time() > deadline:
                return None
            if current_girth >= g:
                # Convert adj to edges
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            
            # Find a short cycle
            # We need to find one to break.
            # get_girth doesn't return the cycle.
            # Let's modify get_girth to return the cycle edges or just one short cycle.
            
            # For now, just pick a random edge and try to swap?
            # No, we should break the shortest cycle.
            
            # Re-implement find_short_cycle that returns the cycle
            def find_short_cycle(n, adj, g):
                # BFS from each node
                for i in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[i] = 0
                    q = deque([i])
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                q.append(v)
                            elif v != parent[u] and u != parent[v]:
                                # Found back edge (u, v)
                                # Reconstruct path
                                path = []
                                cur = u
                                while cur != i:
                                    path.append((cur, parent[cur]))
                                    cur = parent[cur]
                                path.append((u, v))
                                # Check length
                                if len(path) < g:
                                    return path
                return None

            cycle = find_short_cycle(n, adj, g)
            if not cycle:
                return None # Should not happen if current_girth < g

            # Pick a random edge in the cycle to break
            # Let's pick the first edge
            u, v = cycle[0]
            
            # Remove (u, v)
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Find new edges
            # We need to add (u, w) and (v, x) such that (w, x) is an edge?
            # No, 2-switch: Remove (u, v) and (w, x), Add (u, w) and (v, x).
            # We already removed (u, v).
            # Pick a random edge (w, x) in the graph.
            # Ensure w, x are not u, v.
            # Ensure (u, w) and (v, x) are not edges.
            
            found_swap = False
            for _ in range(100): # Try 100 random edges
                # Pick random edge
                w = random.randint(0, n-1)
                x = random.choice(adj[w])
                if w == u or w == v or x == u or x == v:
                    continue
                if (u, w) in zip(adj[u], adj[u]) or (v, x) in zip(adj[v], adj[v]): # Wrong check
                    pass
                # Check if u-w is edge
                if w in adj[u] or x in adj[v]:
                    continue
                # Check if w-x is edge (it is, by selection)
                
                # Perform swap
                adj[u].append(w)
                adj[w].append(u)
                adj[v].append(x)
                adj[x].append(v)
                
                # Remove (w, x)
                adj[w].remove(x)
                adj[x].remove(w)
                
                # Check if graph is still 4-regular (it should be)
                # Check if we created a loop (w==u? no, checked)
                
                new_girth = get_girth(n, adj)
                if new_girth >= current_girth:
                    current_girth = new_girth
                    found_swap = True
                    break
                else:
                    # Revert
                    adj[u].remove(w)
                    adj[w].remove(u)
                    adj[v].remove(x)
                    adj[x].remove(v)
                    adj[w].append(x)
                    adj[x].append(w)
            
            if not found_swap:
                # Revert original removal
                adj[u].append(v)
                adj[v].append(u)
                # Try next iteration or break
                continue

        return None

    if k == 4 and g == 10:
        # Try local search for n < 320
        for n in range(243, 320):
            if time.time() > deadline:
                break
            res = try_local_search(n, g, deadline)
            if res:
                return res

    # Fallback
    return get_luw(k, g)

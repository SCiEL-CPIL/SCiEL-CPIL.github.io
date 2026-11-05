import random
import time
import sys

def construct(k, g):
    # Target: 3-regular graph, girth >= g (16)
    # Strategy: Randomized Local Search (2-opt swaps) on cubic graphs
    # We will try to find a graph with n vertices such that girth >= g.
    # We start with a random cubic graph of size n and try to increase girth.
    # If we fail to reach girth g, we return the best graph found (highest girth, smallest n if valid? No, n must be valid for score).
    # Actually, the score is n. So we want the smallest n that satisfies girth >= g.
    # The evaluator passes k and g. We assume k=3, g=16.
    
    if k != 3:
        # Fallback for other k, though task is specific
        # Use LUW or simple construction
        pass

    deadline = time.time() + 110  # 10s safety margin
    
    # We know n must be even.
    # Moore bound for (3,16) is 512.
    # Known best is 936.
    # Let's search in the range [512, 1024].
    # Since random graphs with n ~ 1000 have girth ~ 10, we might need to start smaller?
    # No, if we start with n=512, girth is likely 6 or 8.
    # If we start with n=1000, girth is likely 10-12.
    # It is easier to increase girth from 12 to 16 than from 6 to 16.
    # So we should start with a larger n and try to shrink? No, n is fixed in the construction.
    # The score is n. We want to minimize n.
    # So we should try n=512, 514, ... and see if we can achieve girth 16.
    # But achieving girth 16 on 512 vertices is extremely hard (it's the cage!).
    # The task says "Smallest known ... 936".
    # So we should aim for n around 900-950.
    
    # Let's try a few candidate sizes near the known best.
    # If we can't find a valid one, we return the LUW result as a fallback?
    # The LUW result is valid but large (~1024).
    # If our search finds a valid graph with n < 1024, we return it.
    # If not, we return the LUW graph.
    
    # First, let's implement the LUW fallback to ensure we have a valid baseline.
    def luw_fallback():
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

    # Try to find a better graph using local search
    # We will try to construct a graph with n vertices and girth >= g.
    # We'll iterate over a range of n values.
    # Since n must be even, step by 2.
    # We know 936 is possible. Let's try n in [900, 1000].
    # If we find one, we return it.
    
    best_n = None
    best_edges = None
    
    # Time budget allocation
    # We have ~110s.
    # Let's try a few n values.
    # Start with n close to 936.
    
    # Helper: Check girth
    def get_girth(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # BFS from each node to find shortest cycle?
        # Efficient girth check:
        # For each edge (u,v), find shortest path from u to v not using (u,v).
        # Girth is min of these paths + 1.
        girth = float('inf')
        for u, v in edges:
            # BFS from u to v, blocking edge (u,v)
            # Since k=3, degree is small.
            # We can do BFS from u, but we must not go to v immediately?
            # Actually, shortest path from u to v in G - (u,v).
            # Let's do BFS from u.
            dist = [-1] * n
            dist[u] = 0
            queue = [u]
            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                if dist[curr] > girth: # Prune
                    break
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        # If nei is v and we are coming from u, skip?
                        # No, BFS finds shortest path.
                        # We just need to ensure we don't use the edge (u,v) directly.
                        # If curr == u and nei == v, skip.
                        if curr == u and nei == v:
                            continue
                        dist[nei] = dist[curr] + 1
                        if nei == v:
                            girth = min(girth, dist[v] + 1)
                            # We can stop this BFS early if we found a short cycle
                            # But we need to check all edges?
                            # Actually, if we find a cycle of length L, girth <= L.
                            # We want to know if girth >= g.
                            # So if we find any cycle < g, we fail.
                            # We can return early if we find a cycle < g.
                            if dist[v] + 1 < g:
                                return dist[v] + 1
                        queue.append(nei)
            # If we didn't return early, no cycle < g found for this edge?
            # Wait, if dist[v] is -1, u and v are disconnected in G-(u,v)? Impossible in connected graph.
            # If dist[v] >= g-1, then cycle length >= g.
            # We must check all edges.
            # Optimization: If we find a cycle of length L < g, return L.
            # If we finish all edges and no cycle < g found, return g (or inf).
        return girth

    # Helper: Generate random cubic graph
    def generate_random_cubic(n):
        # n must be even
        if n % 2 != 0:
            return None
        # Random pairing of 2n half-edges
        half_edges = list(range(n)) * 2
        random.shuffle(half_edges)
        edges = []
        for i in range(0, 2 * n, 2):
            u = half_edges[i]
            v = half_edges[i + 1]
            if u == v:
                continue # Loop, retry? Or just discard and re-pair?
            # Check for multi-edges
            if v in [e[0] for e in edges if e[1] == u] or u in [e[1] for e in edges if e[0] == v]:
                continue
            edges.append((u, v))
        # This simple method might produce loops or multi-edges or disconnected graphs.
        # Better: Use a known algorithm or just try many times.
        # For simplicity, let's use a "wiring" method.
        # Actually, for n=1000, random cubic graphs are usually connected and simple.
        # Let's use a more robust generation:
        # 1. Create list of vertices 0..n-1
        # 2. Shuffle
        # 3. Pair up
        # 4. Check validity
        # 5. If invalid, retry.
        
        for _ in range(100):
            half_edges = list(range(n)) * 2
            random.shuffle(half_edges)
            edges = []
            valid = True
            adj_set = set()
            for i in range(0, 2 * n, 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in adj_set:
                    valid = False
                    break
                adj_set.add((min(u,v), max(u,v)))
                edges.append((u, v))
            if valid and len(edges) == n // 2 * 3: # n/2 * 3? No, n vertices, 3-regular => 3n/2 edges.
                # Check connectivity
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                visited = [False] * n
                visited[0] = True
                queue = [0]
                count = 1
                while queue:
                    curr = queue.pop(0)
                    for nei in adj[curr]:
                        if not visited[nei]:
                            visited[nei] = True
                            count += 1
                            queue.append(nei)
                if count == n:
                    return edges
        return None

    # Local Search
    def local_search(n, g, edges, time_limit):
        # edges is a list of (u,v)
        # Convert to adjacency list for fast lookup
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u,v), max(u,v)))
            
        # Find current girth
        current_girth = get_girth(n, edges)
        if current_girth >= g:
            return edges, current_girth
            
        # Greedy improvement
        # Find all cycles of length < g
        # This is expensive.
        # Instead, let's do random 2-opt swaps and check girth periodically.
        
        start_time = time.time()
        best_edges = edges[:]
        best_girth = current_girth
        
        # We need a fast way to check if a swap is valid and improves girth.
        # Swapping edges (u,v) and (x,y) to (u,x) and (v,y) or (u,y) and (v,x).
        # This is a "2-opt" move.
        
        # To speed up, we can identify "bad" edges (edges in short cycles).
        # But finding short cycles is hard.
        
        # Alternative: Just do random swaps and check girth every K steps.
        # If girth increases, keep. If it decreases, revert with some probability (annealing).
        
        steps = 0
        while time.time() - start_time < time_limit:
            steps += 1
            # Pick two random edges
            if len(edges) < 2:
                break
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            if e1 == e2:
                continue
            u1, v1 = e1
            u2, v2 = e2
            # Check if they share a vertex
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                continue
            
            # Try swap 1: (u1, u2) and (v1, v2)
            # Check if (u1, u2) or (v1, v2) already exists
            if (min(u1,u2), max(u1,u2)) in edge_set or (min(v1,v2), max(v1,v2)) in edge_set:
                # Try other pairing
                if (min(u1,v2), max(u1,v2)) in edge_set or (min(v1,u2), max(v1,u2)) in edge_set:
                    continue
                # Swap to (u1, v2) and (v1, u2)
                new_e1 = (min(u1,v2), max(u1,v2))
                new_e2 = (min(v1,u2), max(v1,u2))
                # Remove old
                edge_set.remove((min(u1,v1), max(u1,v1)))
                edge_set.remove((min(u2,v2), max(u2,v2)))
                # Add new
                edge_set.add(new_e1)
                edge_set.add(new_e2)
                # Update edges list (slow, but we can just update adj)
                # For simplicity, we'll reconstruct edges from edge_set after a batch?
                # No, we need to update adj.
                adj[u1].remove(v1); adj[v1].remove(u1)
                adj[u2].remove(v2); adj[v2].remove(u2)
                adj[u1].append(v2); adj[v2].append(u1)
                adj[v1].append(u2); adj[u2].append(v1)
                
                # Check girth (expensive)
                # Only check every 10 steps or if we have time
                if steps % 5 == 0:
                    g_new = get_girth(n, list(edge_set))
                    if g_new > best_girth:
                        best_girth = g_new
                        best_edges = list(edge_set)
                        if best_girth >= g:
                            return best_edges, best_girth
                    elif g_new < best_girth:
                        # Revert with high probability
                        if random.random() < 0.9:
                            # Revert
                            edge_set.remove(new_e1)
                            edge_set.remove(new_e2)
                            edge_set.add((min(u1,v1), max(u1,v1)))
                            edge_set.add((min(u2,v2), max(u2,v2)))
                            adj[u1].remove(v2); adj[v2].remove(u1)
                            adj[v1].remove(u2); adj[u2].remove(v1)
                            adj[u1].append(v1); adj[v1].append(u1)
                            adj[u2].append(v2); adj[v2].append(u2)
            else:
                # Swap to (u1, u2) and (v1, v2)
                new_e1 = (min(u1,u2), max(u1,u2))
                new_e2 = (min(v1,v2), max(v1,v2))
                edge_set.remove((min(u1,v1), max(u1,v1)))
                edge_set.remove((min(u2,v2), max(u2,v2)))
                edge_set.add(new_e1)
                edge_set.add(new_e2)
                adj[u1].remove(v1); adj[v1].remove(u1)
                adj[u2].remove(v2); adj[v2].remove(u2)
                adj[u1].append(u2); adj[u2].append(u1)
                adj[v1].append(v2); adj[v2].append(v1)
                
                if steps % 5 == 0:
                    g_new = get_girth(n, list(edge_set))
                    if g_new > best_girth:
                        best_girth = g_new
                        best_edges = list(edge_set)
                        if best_girth >= g:
                            return best_edges, best_girth
                    elif g_new < best_girth:
                        if random.random() < 0.9:
                            # Revert
                            edge_set.remove(new_e1)
                            edge_set.remove(new_e2)
                            edge_set.add((min(u1,v1), max(u1,v1)))
                            edge_set.add((min(u2,v2), max(u2,v2)))
                            adj[u1].remove(u2); adj[u2].remove(u1)
                            adj[v1].remove(v2); adj[v2].remove(v1)
                            adj[u1].append(v1); adj[v1].append(u1)
                            adj[u2].append(v2); adj[v2].append(u2)
                            
        return best_edges, best_girth

    # Try a few n values
    # We know 936 is possible.
    # Let's try n = 936, 938, 940, ...
    # If we find a valid graph, return it.
    
    # To save time, we will only try a few n values.
    # Start with n=936.
    for n in [936, 938, 940, 942, 944, 946, 948, 950]:
        if time.time() > deadline - 10:
            break
        edges = generate_random_cubic(n)
        if not edges:
            continue
        
        # Run local search
        time_left = deadline - time.time() - 5
        if time_left < 10:
            break
        found_edges, girth = local_search(n, g, edges, time_left)
        if girth >= g:
            return n, found_edges
    
    # Fallback to LUW
    return luw_fallback()

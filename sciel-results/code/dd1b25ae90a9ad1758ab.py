import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return time.time() > deadline
    
    # Moore bound for k-regular graph of girth g
    def moore_bound(k, g):
        if g % 2 == 1:  # odd girth
            d = (g - 1) // 2
            return k * (k - 1) ** d + 2 * sum((k - 1) ** i for i in range(d))
        else:  # even girth
            d = g // 2
            return 2 * sum((k - 1) ** i for i in range(d))
    
    # Fast girth check using BFS from each vertex
    def find_girth(adj, n, target_g):
        """Find girth of graph. Returns girth or inf if >= target_g."""
        min_girth = float('inf')
        for start in range(n):
            if check_deadline():
                return min_girth
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        q.append(w)
                    elif w != start and parent[w] != u and parent[u] != w:
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < target_g:
                            return min_girth
        return min_girth
    
    # Generate a random k-regular graph using configuration model
    def random_k_regular(n, k, rng):
        """Generate random k-regular graph using configuration model."""
        if n * k % 2 != 0:
            return None
        # Create stubs
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        rng.shuffle(stubs)
        
        # Pair up stubs to form edges
        adj = [[] for _ in range(n)]
        edge_set = set()
        i = 0
        while i < len(stubs):
            u = stubs[i]
            j = i + 1
            # Find a partner
            while j < len(stubs):
                v = stubs[j]
                if u != v and (min(u,v), max(u,v)) not in edge_set:
                    break
                j += 1
            if j >= len(stubs):
                return None  # Failed to create valid graph
            v = stubs[j]
            edge_set.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
            i = j + 1
        
        # Verify regularity
        for v in range(n):
            if len(adj[v]) != k:
                return None
        return adj
    
    # Local search to improve girth
    def improve_girth(adj, n, k, target_g, max_iter=5000, rng=None):
        """Try to improve girth by local edge swaps."""
        if rng is None:
            rng = random.Random(42)
        
        # Find all short cycles
        def find_short_cycles(adj, n, max_len):
            """Find all cycles of length <= max_len."""
            cycles = []
            for start in range(n):
                if check_deadline():
                    break
                # BFS to find short cycles
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_len // 2:
                        continue
                    for w in adj[u]:
                        if dist[w] == -1:
                            dist[w] = dist[u] + 1
                            parent[w] = u
                            q.append(w)
                        elif w != start and parent[w] != u and parent[u] != w:
                            cycle_len = dist[u] + dist[w] + 1
                            if cycle_len <= max_len:
                                # Reconstruct cycle
                                cycle = [u]
                                cur = u
                                while cur != start:
                                    cur = parent[cur]
                                    cycle.append(cur)
                                cycle2 = [w]
                                cur = w
                                while cur != start:
                                    cur = parent[cur]
                                    cycle2.append(cur)
                                cycle.extend(reversed(cycle2[1:]))
                                cycles.append(cycle)
            return cycles
        
        current_girth = find_girth(adj, n, target_g)
        if current_girth >= target_g:
            return adj, current_girth
        
        for iteration in range(max_iter):
            if check_deadline():
                break
            
            # Find a short cycle
            cycles = find_short_cycles(adj, n, min(target_g - 1, 8))
            if not cycles:
                current_girth = find_girth(adj, n, target_g)
                if current_girth >= target_g:
                    break
                continue
            
            cycle = cycles[0]
            cycle_len = len(cycle)
            
            # Try to break this cycle by swapping edges
            # Pick two edges in the cycle and try to swap
            # Edge (cycle[i], cycle[i+1]) can be swapped with some other edge
            
            # Try all pairs of edges in the cycle
            improved = False
            for i in range(cycle_len):
                if check_deadline():
                    break
                u1 = cycle[i]
                v1 = cycle[(i + 1) % cycle_len]
                
                # Try to swap edge (u1, v1) with another edge
                for u2 in adj[u1]:
                    if u2 == v1:
                        continue
                    for v2 in adj[v1]:
                        if v2 == u1 or v2 == u2:
                            continue
                        # Check if (u2, v2) is an edge
                        if v2 not in adj[u2] and u2 not in adj[v2]:
                            continue
                        
                        # Check if swap is valid (no multi-edges, no loops)
                        # Remove (u1,v1) and (u2,v2), add (u1,v2) and (u2,v1)
                        # Check validity
                        if v2 in adj[u1] or v1 in adj[u2] or u2 in adj[v1] or u1 in adj[v2]:
                            continue
                        
                        # Perform swap
                        adj[u1].remove(v1)
                        adj[v1].remove(u1)
                        if u2 in adj[u1]:
                            adj[u1].remove(u2)
                        if v2 in adj[v1]:
                            adj[v1].remove(v2)
                        adj[u1].append(v2)
                        adj[v2].append(u1)
                        adj[u2].append(v1)
                        adj[v1].append(u2)
                        
                        # Check if this improves girth
                        new_girth = find_girth(adj, n, target_g)
                        if new_girth > current_girth:
                            current_girth = new_girth
                            improved = True
                            break
                        else:
                            # Undo swap
                            adj[u1].remove(v2)
                            adj[v2].remove(u1)
                            adj[u2].remove(v1)
                            adj[v1].remove(u2)
                            adj[u1].append(v1)
                            adj[v1].append(u1)
                            adj[u2].append(u2)  # placeholder
                            adj[u2].remove(u2)
                            adj[v2].append(v2)  # placeholder
                            adj[v2].remove(v2)
                            # Restore properly
                            if v1 not in adj[u1]:
                                adj[u1].append(v1)
                            if u1 not in adj[v1]:
                                adj[v1].append(u1)
                            if v2 not in adj[u2]:
                                adj[u2].append(v2)
                            if u2 not in adj[v2]:
                                adj[v2].append(u2)
                        if improved:
                            break
                if improved:
                    break
            
            if current_girth >= target_g:
                break
        
        return adj, current_girth
    
    # Voltage graph construction
    def voltage_graph_construct(k, g):
        """Construct k-regular graph using voltage graph over K_{k,k}."""
        best_n = None
        best_edges = None
        rng = random.Random(42)
        
        # Try different values of m
        for m in range(30, 300, 5):
            if check_deadline():
                break
            
            n = 2 * k * m
            # Try several random voltage assignments
            num_attempts = max(5, min(50, 2000 // m))
            
            for attempt in range(num_attempts):
                if check_deadline():
                    break
                
                # Generate random voltages
                voltages = [[rng.randint(0, m-1) for _ in range(k)] for _ in range(k)]
                
                # Build the graph
                adj = [[] for _ in range(n)]
                edges = []
                
                for i in range(k):
                    for j in range(k):
                        v = voltages[i][j]
                        for a in range(m):
                            b = (a + v) % m
                            u = i * m + a
                            w = k * m + j * m + b
                            if w not in adj[u]:
                                edges.append((u, w))
                                adj[u].append(w)
                                adj[w].append(u)
                
                # Check girth
                girth = find_girth(adj, n, g)
                
                if girth >= g:
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                        if n < 1000:
                            return best_n, best_edges
        
        return best_n, best_edges
    
    # Try voltage graph construction first
    n_vg, edges_vg = voltage_graph_construct(k, g)
    
    if n_vg is not None and edges_vg is not None:
        # Verify the graph
        deg = [0] * n_vg
        for u, v in edges_vg:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg):
            return n_vg, edges_vg
    
    # Fallback: Random construction + local search
    rng = random.Random(123)
    best_n = None
    best_edges = None
    
    # Try different sizes
    for target_n in range(600, 1200, 50):
        if check_deadline():
            break
        
        for attempt in range(3):
            if check_deadline():
                break
            
            adj = random_k_regular(target_n, k, rng)
            if adj is None:
                continue
            
            # Try to improve girth
            adj, girth = improve_girth(adj, target_n, k, g, max_iter=2000, rng=rng)
            
            if girth >= g:
                edges = []
                for u in range(target_n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                if best_n is None or target_n < best_n:
                    best_n = target_n
                    best_edges = edges
                break
    
    if best_n is not None and best_edges is not None:
        return best_n, best_edges
    
    # Final fallback: LUW construction
    def luw_construct(k, g):
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
    
    n_luw, edges_luw = luw_construct(k, g)
    return n_luw, edges_luw

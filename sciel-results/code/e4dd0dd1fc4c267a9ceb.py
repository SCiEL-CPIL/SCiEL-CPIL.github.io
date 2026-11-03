import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 115.0
    
    # For 3-regular graph with girth >= 16
    # Moore bound: 2 * (3^(g/2 - 1) - 1) = 2 * (3^7 - 1) = 2 * 2186 = 4372? No wait
    # For girth g (even), Moore bound is 2 * sum_{i=0}^{g/2-1} (k-1)^i
    # For k=3, g=16: 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254
    # Actually the Moore bound for (3,16) cage is 2*(3^7 - 1)/2 = 3^7 - 1 = 2186? No.
    # The Moore bound: n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^{g/2-1})
    # For k=3, g=16: 2 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 2 * 127 = 254
    # But the problem says best lower bound is 512. Let me re-read.
    # "The smallest known such graph has 936 vertices... and the best lower bound is 512."
    # So Moore bound gives 254, but a better lower bound is 512.
    
    # We'll try to find a graph with n vertices where n is even, >= 254
    # Start with a reasonable n and use local search
    
    def generate_random_3_regular(n):
        """
        Generates a random 3-regular graph on n vertices using configuration model.
        Returns (adj, edges) where adj is list of sets, edges is list of (u,v) pairs.
        """
        if n % 2 != 0:
            n += 1
            
        for attempt in range(100):
            stubs = [i for i in range(n) for _ in range(3)]
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            edges = []
            valid = True
            for i in range(0, 3*n, 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
                edges.append((min(u, v), max(u, v)))
            if valid:
                return n, adj, edges
        return None
    
    def find_shortest_cycle(n, adj):
        """
        Find the length of the shortest cycle in the graph.
        Returns (girth, cycle) where cycle is a list of vertices forming the cycle, or None.
        """
        best_girth = float('inf')
        best_cycle = None
        
        for start in range(n):
            # BFS to find shortest cycle through start
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and dist[u] + dist[v] + 1 < best_girth:
                        # Found a cycle
                        # Trace back from u and v to find LCA
                        cycle = []
                        # Path from u up to LCA
                        path_u = []
                        curr = u
                        while curr != -1:
                            path_u.append(curr)
                            curr = parent[curr]
                        # Path from v up to LCA  
                        path_v = []
                        curr = v
                        while curr != -1:
                            path_v.append(curr)
                            curr = parent[curr]
                        
                        # Find LCA
                        set_u = set(path_u)
                        lca = None
                        for node in path_v:
                            if node in set_u:
                                lca = node
                                break
                        
                        if lca is not None:
                            # Build cycle
                            idx_u = path_u.index(lca)
                            idx_v = path_v.index(lca)
                            cycle_part_u = path_u[:idx_u+1]  # from u to lca
                            cycle_part_v = path_v[:idx_v+1]  # from v to lca
                            cycle = cycle_part_u + cycle_part_v[::-1]
                            cycle_len = len(cycle)
                            if cycle_len < best_girth:
                                best_girth = cycle_len
                                best_cycle = cycle
                                if best_girth <= g:
                                    return best_girth, best_cycle
                        break  # Only need to check first cycle found per start for efficiency
        
        return best_girth, best_cycle
    
    def try_edge_swap(n, adj, edges, edge_to_remove, new_u, new_v):
        """
        Try swapping an edge. Returns True if swap is valid (no loops, no multi-edges).
        Modifies adj in place.
        """
        u, v = edge_to_remove
        # Check if new edge would be valid
        if new_u == new_v:
            return False
        if new_v in adj[new_u]:
            return False
        
        # Perform swap
        adj[u].discard(v)
        adj[v].discard(u)
        adj[new_u].add(new_v)
        adj[new_v].add(new_u)
        
        return True
    
    def undo_edge_swap(n, adj, edge_to_remove, new_u, new_v):
        """
        Undo an edge swap.
        """
        u, v = edge_to_remove
        adj[u].add(v)
        adj[v].add(u)
        adj[new_u].discard(new_v)
        adj[new_v].discard(new_u)
    
    # Main search
    best_result = None
    
    # Try different starting sizes
    for n in [254, 256, 260, 264, 272, 280, 288, 300, 320, 340, 360, 400, 440, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 936]:
        if time.time() - start_time > time_limit - 5:
            break
        
        result = generate_random_3_regular(n)
        if result is None:
            continue
        
        n_actual, adj, edges = result
        
        # Local search to increase girth
        for iteration in range(500):
            if time.time() - start_time > time_limit - 5:
                break
            
            girth, cycle = find_shortest_cycle(n_actual, adj)
            
            if girth >= g:
                # Found a valid graph!
                edge_list = []
                for u in range(n_actual):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append((u, v))
                if best_result is None or n_actual < best_result[0]:
                    best_result = (n_actual, edge_list)
                break
            
            if cycle is None:
                break
            
            # Try to break the shortest cycle by swapping an edge
            # Pick a random edge in the cycle and try to swap it
            if len(cycle) < 3:
                break
                
            # Try all edges in the cycle
            improved = False
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                if u < v:
                    cycle_edges.append((u, v))
                else:
                    cycle_edges.append((v, u))
            
            random.shuffle(cycle_edges)
            
            for edge in cycle_edges:
                u, v = edge
                # Pick a random neighbor of u (not v) and a random neighbor of v (not u)
                neighbors_u = [x for x in adj[u] if x != v]
                neighbors_v = [x for x in adj[v] if x != u]
                
                if not neighbors_u or not neighbors_v:
                    continue
                
                new_u = random.choice(neighbors_u)
                new_v = random.choice(neighbors_v)
                
                if try_edge_swap(n_actual, adj, edges, edge, new_u, new_v):
                    # Check if this improved the girth
                    new_girth, new_cycle = find_shortest_cycle(n_actual, adj)
                    if new_girth > girth:
                        # Keep the swap
                        improved = True
                        # Update edges list
                        edge_list = []
                        for uu in range(n_actual):
                            for vv in adj[uu]:
                                if uu < vv:
                                    edge_list.append((uu, vv))
                        edges = edge_list
                        break
                    else:
                        # Undo the swap
                        undo_edge_swap(n_actual, adj, edge, new_u, new_v)
            
            if not improved:
                # Try a random edge swap not in the cycle
                all_edges = []
                for u in range(n_actual):
                    for v in adj[u]:
                        if u < v:
                            all_edges.append((u, v))
                
                if not all_edges:
                    break
                
                random.shuffle(all_edges)
                for edge in all_edges[:20]:  # Try a few random edges
                    u, v = edge
                    neighbors_u = [x for x in adj[u] if x != v]
                    neighbors_v = [x for x in adj[v] if x != u]
                    
                    if not neighbors_u or not neighbors_v:
                        continue
                    
                    new_u = random.choice(neighbors_u)
                    new_v = random.choice(neighbors_v)
                    
                    if try_edge_swap(n_actual, adj, edges, edge, new_u, new_v):
                        new_girth, new_cycle = find_shortest_cycle(n_actual, adj)
                        if new_girth >= girth:
                            # Keep if not worse
                            edge_list = []
                            for uu in range(n_actual):
                                for vv in adj[uu]:
                                    if uu < vv:
                                        edge_list.append((uu, vv))
                            edges = edge_list
                            break
                        else:
                            undo_edge_swap(n_actual, adj, edge, new_u, new_v)
        
        if best_result is not None and best_result[0] <= n:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a known valid construction
    # Use the LUW-style construction from the original as fallback
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

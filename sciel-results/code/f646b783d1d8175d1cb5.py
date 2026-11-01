import time
import random
import sys
import numpy as np

def construct(k, g):
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
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    if k == 4 and g == 10:
        # Try to construct a voltage graph lift of the Petersen graph
        # with group Z_32 to get n=320.
        
        # Petersen graph vertices 0..9
        # Edges:
        # Outer pentagon: 0-1, 1-2, 2-3, 3-4, 4-0
        # Inner star: 5-7, 7-9, 9-6, 6-8, 8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        # Map edge to index for voltage assignment
        edge_to_idx = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
            
        # We need to assign voltages w[i] in Z_32
        # Check all closed walks of length < 10
        # A closed walk is a sequence of vertices v0, v1, ..., vL=v0
        # Sum of voltages along the walk must be != 0 mod 32
        
        # Precompute all closed walks of length 5 to 9
        # Since Petersen is small, we can enumerate them.
        
        closed_walks = []
        for L in range(5, 10):
            # Generate all walks of length L starting from each vertex
            # and check if they are closed
            # Use DFS or iterative deepening
            # Given small size, simple recursion is fine
            
            def gen_walks(start, curr, length, path):
                if length == L:
                    if curr == start:
                        closed_walks.append(path)
                    return
                for u, v in pet_edges:
                    if curr == u:
                        gen_walks(start, v, length + 1, path + [(u, v)])
                    elif curr == v:
                        gen_walks(start, u, length + 1, path + [(v, u)])
                        
            for s in range(10):
                gen_walks(s, s, 0, [])
                
        # Remove duplicates? 
        # Different walks might have same edge sequence but different start?
        # Actually, we just need to check the sum of voltages.
        # A walk is defined by the sequence of edges.
        # We can represent a walk by the sequence of edge indices.
        
        # Let's re-generate as sequences of edge indices
        closed_walk_edges = set()
        for L in range(5, 10):
            def gen_walk_edges(start, curr, length, path):
                if length == L:
                    if curr == start:
                        closed_walk_edges.add(tuple(path))
                    return
                for idx, (u, v) in enumerate(pet_edges):
                    if curr == u:
                        gen_walk_edges(start, v, length + 1, path + [idx])
                    elif curr == v:
                        gen_walk_edges(start, u, length + 1, path + [idx])
                        
            for s in range(10):
                gen_walk_edges(s, s, 0, [])
                
        # Now we have a set of tuples of edge indices.
        # For each tuple, the sum of voltages (with signs) must be != 0.
        # Wait, the voltage on edge (u,v) is w[idx].
        # If we traverse (v,u), the voltage is -w[idx].
        # So we need to track direction.
        
        # Let's re-do with direction.
        closed_walks_dir = set()
        for L in range(5, 10):
            def gen_walk_dir(start, curr, length, path):
                if length == L:
                    if curr == start:
                        closed_walks_dir.add(tuple(path))
                    return
                for idx, (u, v) in enumerate(pet_edges):
                    if curr == u:
                        gen_walk_dir(start, v, length + 1, path + [(idx, 1)]) # +w[idx]
                    elif curr == v:
                        gen_walk_dir(start, u, length + 1, path + [(idx, -1)]) # -w[idx]
                        
            for s in range(10):
                gen_walk_dir(s, s, 0, [])
                
        # Search for voltage assignment
        deadline = time.time() + 100
        found = False
        best_voltages = None
        
        # Random search
        for _ in range(100000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, 31) for _ in range(15)]
            valid = True
            for walk in closed_walks_dir:
                total = 0
                for idx, sign in walk:
                    total += sign * voltages[idx]
                if total % 32 == 0:
                    valid = False
                    break
            if valid:
                best_voltages = voltages
                found = True
                break
                
        if found:
            # Construct the lift
            n = 10 * 32
            edges = []
            # Vertices are (v, g) where v in 0..9, g in 0..31
            # Index = v * 32 + g
            for u, v in pet_edges:
                idx = edge_to_idx[(u, v)]
                w = best_voltages[idx]
                for g in range(32):
                    u_idx = u * 32 + g
                    v_idx = v * 32 + (g + w) % 32
                    edges.append((u_idx, v_idx))
                    
            # Verify girth
            # Build adj list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Check girth >= 10
            def has_girth_at_least(n, edges, g):
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Check for simple graph
                edge_set = set()
                for u, v in edges:
                    if u == v:
                        return False
                    if (v, u) in edge_set:
                        return False
                    edge_set.add((u, v))
                
                # BFS from each vertex
                for start in range(n):
                    dist = [-1] * n
                    parent = [-1] * n
                    dist[start] = 0
                    queue = [start]
                    head = 0
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        if dist[u] >= g // 2:
                            continue
                        for v in adj[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                parent[v] = u
                                queue.append(v)
                            else:
                                if v != parent[u] and u != parent[v]:
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < g:
                                        return False
                return True
                
            if has_girth_at_least(n, edges, g):
                return n, edges
            else:
                # Should not happen if voltage search was correct
                # But maybe we missed some cycles?
                # Fallback to LUW
                pass

    # Fallback to LUW
    return luw_construct(k, g)

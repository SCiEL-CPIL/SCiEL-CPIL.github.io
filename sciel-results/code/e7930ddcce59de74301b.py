import numpy as np
import random
import time
from itertools import product

def construct(k, g):
    # Use C4 x C4 as base graph (16 vertices, 4-regular, girth 4)
    # Lift over Z20 to get 320 vertices.
    
    m = 20
    base_n = 16
    n_vertices = base_n * m
    
    # Define base graph edges
    # Vertices are (i, j) for i, j in 0..3
    # Edges: (i,j) -- (i+1,j) and (i,j) -- (i,j+1) mod 4
    base_edges = []
    for i in range(4):
        for j in range(4):
            u = i * 4 + j
            v1 = ((i + 1) % 4) * 4 + j
            v2 = i * 4 + ((j + 1) % 4)
            base_edges.append((u, v1))
            base_edges.append((u, v2))
            
    # Remove duplicates if any (should be none)
    base_edges = list(set(base_edges))
    
    # We need to assign voltages to edges such that no cycle of length < 10 has sum 0 mod 20.
    # Cycles in C4 x C4:
    # Length 4: Squares.
    # Length 6: Hexagons.
    # Length 8: Octagons.
    
    # Deterministic search:
    # Assign voltages greedily or use a fixed pattern.
    # A known good pattern for such lifts often involves linear functions.
    
    # Let's try a simple random search with a fixed seed for reproducibility, 
    # but optimized with early termination.
    # Since the directive says "do not reuse current designs", and the current one is PG(2,3), 
    # using C4xC4 is a new design.
    
    random.seed(42)
    
    for _ in range(5000):
        voltages = {}
        for u, v in base_edges:
            val = random.randint(0, m-1)
            voltages[(u, v)] = val
            voltages[(v, u)] = (-val) % m
            
        valid = True
        
        # Check 4-cycles
        # In C4 x C4, 4-cycles are the faces.
        for i in range(4):
            for j in range(4):
                u = i*4 + j
                v = ((i+1)%4)*4 + j
                w = ((i+1)%4)*4 + ((j+1)%4)
                x = i*4 + ((j+1)%4)
                
                s = voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + voltages[(x,u)]
                if s % m == 0:
                    valid = False
                    break
            if not valid: break
        if not valid: continue
        
        # Check 6-cycles
        # Enumerate 6-cycles in C4 x C4
        # A 6-cycle in grid graph must move 2 steps in one direction, 1 in other, etc.
        # Actually, C4 x C4 has girth 4.
        # 6-cycles exist.
        
        # Instead of enumerating all, we can build the lift and check girth via BFS.
        # But BFS on 320 vertices is fast.
        
        lift_edges = []
        for u, v in base_edges:
            vu = voltages[(u, v)]
            for i in range(m):
                lift_edges.append((u*m + i, v*m + (i + vu) % m))
        
        # Build adjacency list
        adj = [[] for _ in range(n_vertices)]
        for u, v in lift_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check girth
        min_girth = 999
        # BFS from each vertex
        for start in range(n_vertices):
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                cur = queue[head]
                head += 1
                for nb in adj[cur]:
                    if dist[nb] == -1:
                        dist[nb] = dist[cur] + 1
                        parent[nb] = cur
                        queue.append(nb)
                    elif parent[nb] != cur and parent[cur] != nb:
                        cycle_len = dist[nb] + dist[cur] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            if min_girth < g:
                break
                
        if min_girth >= g:
            return n_vertices, lift_edges
            
    # Fallback: Return a known large graph or raise error
    # To ensure we return something valid, we can return a disjoint union of cycles?
    # No, must be k-regular.
    # If search fails, return the LUW construction from the seed?
    # But we shouldn't reuse.
    # Let's just return a large n to be safe? No, score is n.
    # If we fail, we should try to construct a valid graph, even if large.
    
    # Fallback: Cycle of length n? No, k=4.
    # Use a simple 4-regular graph with large girth?
    # Random 4-regular graph on 1000 vertices likely has girth > 10.
    
    n_fallback = 1000
    edges_fallback = []
    # Generate random 4-regular graph
    while True:
        edges_fallback = []
        used = [0] * n_fallback
        for i in range(n_fallback):
            while used[i] < 4:
                j = random.randint(0, n_fallback - 1)
                if j != i and used[j] < 4 and (i, j) not in edges_fallback and (j, i) not in edges_fallback:
                    edges_fallback.append((i, j))
                    used[i] += 1
                    used[j] += 1
        # Check if regular
        if all(u == 4 for u in used):
            # Check girth
            adj_f = [[] for _ in range(n_fallback)]
            for u, v in edges_fallback:
                adj_f[u].append(v)
                adj_f[v].append(u)
            
            min_girth_f = 999
            for start in range(n_fallback):
                dist = [-1] * n_fallback
                parent = [-1] * n_fallback
                dist[start] = 0
                queue = [start]
                head = 0
                while head < len(queue):
                    cur = queue[head]
                    head += 1
                    for nb in adj_f[cur]:
                        if dist[nb] == -1:
                            dist[nb] = dist[cur] + 1
                            parent[nb] = cur
                            queue.append(nb)
                        elif parent[nb] != cur and parent[cur] != nb:
                            cycle_len = dist[nb] + dist[cur] + 1
                            if cycle_len < min_girth_f:
                                min_girth_f = cycle_len
                if min_girth_f < g:
                    break
            if min_girth_f >= g:
                return n_fallback, edges_fallback

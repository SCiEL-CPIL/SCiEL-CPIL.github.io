"""Spectral C4 minimization for 31 vertices, 91 edges."""
import numpy as np
import random
import time

def construct():
    n = 31
    m = 91
    
    start_time = time.time()
    timeout = 115
    
    def count_c4s(adj):
        """Count C4s using common neighbor pairs."""
        c4 = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    c4 += common * (common - 1) // 2
        return c4
    
    def initial_graph(seed):
        """Create a structured initial graph with low C4 count."""
        random.seed(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        
        # Start with a random spanning tree
        order = list(range(n))
        random.shuffle(order)
        for i in range(1, n):
            u = order[i]
            v = order[random.randint(0, i - 1)]
            edge = (min(u, v), max(u, v))
            edges.add(edge)
            adj[u].add(v)
            adj[v].add(u)
        
        # Add remaining edges greedily, preferring edges with few common neighbors
        remaining = m - (n - 1)
        attempts = 0
        max_attempts = 50000
        while len(edges) < m and attempts < max_attempts:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                attempts += 1
                continue
            edge = (min(u, v), max(u, v))
            if edge in edges:
                attempts += 1
                continue
            
            # Count common neighbors
            common = len(adj[u] & adj[v])
            # Accept if common neighbors is low
            if common <= 1 or (common == 2 and random.random() < 0.3):
                edges.add(edge)
                adj[u].add(v)
                adj[v].add(u)
            attempts += 1
        
        # If still short, add random edges
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                edge = (min(u, v), max(u, v))
                if edge not in edges:
                    edges.add(edge)
                    adj[u].add(v)
                    adj[v].add(u)
        
        return edges, adj
    
    best_edges = None
    best_c4 = float('inf')
    
    # Try multiple initializations with different seeds
    for init_seed in range(8):
        if time.time() - start_time > timeout:
            break
        
        edges, adj = initial_graph(init_seed)
        c4 = count_c4s(adj)
        
        if c4 < best_c4:
            best_c4 = c4
            best_edges = set(edges)
        
        # Local search: try edge swaps to reduce C4
        improved = True
        while improved and (time.time() - start_time) < timeout:
            improved = False
            edge_list = list(edges)
            random.shuffle(edge_list)
            
            for (a, b) in edge_list:
                if time.time() - start_time > timeout:
                    break
                
                # Compute C4 delta from removing edge (a, b)
                delta_rem = 0
                for x in adj[b]:
                    if x != a:
                        delta_rem -= (len(adj[b] & adj[x]) - 1)
                for y in adj[a]:
                    if y != b:
                        delta_rem -= (len(adj[a] & adj[y]) - 1)
                
                # Try to find a beneficial swap
                found = False
                for _ in range(30):
                    c = random.randint(0, n - 1)
                    d = random.randint(0, n - 1)
                    if c == d:
                        continue
                    c, d = min(c, d), max(c, d)
                    if (c, d) in edges or (c == a and d == b):
                        continue
                    
                    # Compute C4 delta from adding edge (c, d)
                    delta_add = 0
                    for x in adj[d]:
                        if x != c:
                            delta_add += len(adj[c] & adj[x])
                    for y in adj[c]:
                        if y != d:
                            delta_add += len(adj[d] & adj[y])
                    
                    total_delta = delta_rem + delta_add
                    if total_delta < 0:
                        # Perform swap
                        edges.remove((a, b))
                        edges.add((c, d))
                        adj[a].remove(b)
                        adj[b].remove(a)
                        adj[c].add(d)
                        adj[d].add(c)
                        c4 += total_delta
                        improved = True
                        found = True
                        break
                
                if found:
                    continue
        
        if c4 < best_c4:
            best_c4 = c4
            best_edges = set(edges)
    
    result = [[u, v] for u, v in sorted(best_edges)]
    return result

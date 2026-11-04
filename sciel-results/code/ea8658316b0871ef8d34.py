"""Refined program for 4-regular girth 10 cage construction."""
# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    deadline = time.time() + 130.0
    random.seed(42)
    
    n = 320
    
    def generate_random_regular_graph(n, degree):
        stubs = list(range(n)) * degree
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u,v), max(u,v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        
        if not valid or len(edges) != n * degree // 2:
            return None
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def compute_girth(adj, n):
        """Compute exact girth."""
        min_girth = 2 * n + 1
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = [start]
            dist[start] = 0
            idx = 0
            while idx < len(queue):
                u = queue[idx]
                idx += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth
    
    # Try to find a graph with girth >= g
    best_adj = None
    best_girth = 0
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        adj = generate_random_regular_graph(n, k)
        if adj is None:
            continue
        
        girth = compute_girth(adj, n)
        if girth > best_girth:
            best_girth = girth
            best_adj = adj
        
        if girth >= g:
            break
    
    if best_adj is None:
        adj = generate_random_regular_graph(n, k)
        best_adj = adj
    
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append((u, v))
    
    return n, edges
# EVOLVE-BLOCK-END

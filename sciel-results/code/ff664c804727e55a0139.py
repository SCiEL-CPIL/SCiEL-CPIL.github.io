import time
import random
from itertools import combinations

def _check_validity(edges, n):
    """Check if graph is valid: no 3-path with equal-degree endpoints."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    
    for u in range(n):
        for v in adj[u]:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def construct():
    n = 15
    deadline = time.time() + 80
    best_edges = []
    best_count = 0
    rng = random.Random(42)
    
    all_edges = list(combinations(range(n), 2))
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        edge_order = all_edges[:]
        rng.shuffle(edge_order)
        
        adj = [set() for _ in range(n)]
        edges = []
        
        for u, v in edge_order:
            adj[u].add(v)
            adj[v].add(u)
            
            deg = [len(adj[i]) for i in range(n)]
            valid = True
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                adj[u].discard(v)
                adj[v].discard(u)
            else:
                edges.append([u, v])
        
        # Local search: add any possible edge
        current_edge_set = set(tuple(sorted(e)) for e in edges)
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                e_key = (min(u, v), max(u, v))
                if e_key in current_edge_set:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                deg = [len(adj[i]) for i in range(n)]
                valid = True
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    edges.append([u, v])
                    current_edge_set.add(e_key)
                    improved = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [e[:] for e in edges]
    
    # Final validity check
    if best_edges and not _check_validity(best_edges, n):
        # Fallback to star
        best_edges = [[0, v] for v in range(1, n)]
    
    return [[int(u), int(v)] for u, v in best_edges]

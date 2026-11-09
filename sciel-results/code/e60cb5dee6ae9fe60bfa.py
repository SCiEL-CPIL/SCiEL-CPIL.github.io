import random
import itertools
from collections import defaultdict

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_triangles(n, adj):
    """Find all triangles in the graph."""
    triangles = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        triangles.append((u, v, w))
    return triangles

def _max_triangle_degree_sum(n, adj):
    """Compute the maximum degree sum over all triangles."""
    if n <= 2:
        return 0
    triangles = _find_triangles(n, adj)
    if not triangles:
        return 0
    max_sum = 0
    for (u, v, w) in triangles:
        s = len(adj[u]) + len(adj[v]) + len(adj[w])
        if s > max_sum:
            max_sum = s
    return max_sum

def construct():
    n = 16
    target_edges = 65
    
    # Initial state: K_{8,8} - (0, 8) + (0, 1)
    # This gives score 25.
    def initial_graph():
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                if u == 0 and v == 8:
                    continue
                edges.add(frozenset((u, v)))
        edges.add(frozenset((0, 1)))
        return [list(e) for e in edges]
    
    best_score = float('inf')
    best_edges = None
    
    # Run multiple restarts
    for restart in range(20):
        random.seed(42 + restart)
        edges = initial_graph()
        adj = _build_adjacency(n, edges)
        current_score = _max_triangle_degree_sum(n, adj)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = edges[:]
            
        # Hill Climbing
        max_iters = 5000
        for it in range(max_iters):
            edges_set = set(frozenset(e) for e in edges)
            existing = list(edges_set)
            if not existing:
                break
            
            # Try a random swap
            e_remove = random.choice(existing)
            u, v = list(e_remove)
            
            all_pairs = [(a, b) for a in range(n) for b in range(a+1, n)]
            candidates = []
            for a, b in all_pairs:
                if frozenset((a, b)) == e_remove:
                    continue
                if frozenset((a, b)) in edges_set:
                    continue
                candidates.append((a, b))
            
            if not candidates:
                break
            
            a, b = random.choice(candidates)
            
            new_edges_set = set(edges_set)
            new_edges_set.remove(e_remove)
            new_edges_set.add(frozenset((a, b)))
            
            if len(new_edges_set) != target_edges:
                continue
            
            new_adj = _build_adjacency(n, [list(e) for e in new_edges_set])
            new_score = _max_triangle_degree_sum(n, new_adj)
            
            if new_score < current_score:
                current_score = new_score
                edges = [list(e) for e in new_edges_set]
                adj = new_adj
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges[:]
            else:
                # No improvement, continue (Hill Climbing)
                pass
        
        if best_score <= 24:
            break
            
    return {'n': n, 'edges': best_edges}

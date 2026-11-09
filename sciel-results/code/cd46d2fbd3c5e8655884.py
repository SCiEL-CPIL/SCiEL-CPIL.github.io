import random
import itertools
import math
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
    
    random.seed(42)
    
    def initial_graph():
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                edges.add(frozenset((u, v)))
        # Add one edge within part A to reach 65 edges
        edges.add(frozenset((0, 1)))
        return [list(e) for e in edges]
    
    best_score = float('inf')
    best_edges = None
    
    # Run multiple independent local searches
    num_restarts = 10
    
    for restart in range(num_restarts):
        edges = initial_graph()
        if restart > 0:
            # Perturb the initial graph slightly to diversify
            edges_set = set(frozenset(e) for e in edges)
            for _ in range(5):
                # Random swap
                existing = list(edges_set)
                if not existing: continue
                e_remove = random.choice(existing)
                u, v = list(e_remove)
                # Find a non-edge to add
                all_pairs = [(a, b) for a in range(n) for b in range(a+1, n)]
                candidates = [(a,b) for a,b in all_pairs if frozenset((a,b)) not in edges_set and frozenset((a,b)) != e_remove]
                if candidates:
                    a, b = random.choice(candidates)
                    edges_set.remove(e_remove)
                    edges_set.add(frozenset((a,b)))
            edges = [list(e) for e in edges_set]
        
        adj = _build_adjacency(n, edges)
        current_score = _max_triangle_degree_sum(n, adj)
        
        temp = 5.0
        cooling = 0.995
        min_temp = 0.01
        max_iters = 20000
        
        for it in range(max_iters):
            if temp < min_temp:
                break
            
            edges_set = set(frozenset(e) for e in edges)
            
            # Targeted Move: Pick an edge from the worst triangle to remove
            triangles = _find_triangles(n, adj)
            if not triangles:
                break # No triangles, score 0, optimal
            
            # Find worst triangle
            worst_tri = max(triangles, key=lambda t: len(adj[t[0]]) + len(adj[t[1]]) + len(adj[t[2]]))
            u, v, w = worst_tri
            # Remove one edge from this triangle
            e_remove = random.choice([frozenset((u, v)), frozenset((v, w)), frozenset((u, w))])
            
            # Find candidates to add
            # Prefer edges that don't create new high-degree triangles
            all_pairs = [(a, b) for a in range(n) for b in range(a+1, n)]
            candidates = []
            for a, b in all_pairs:
                key = frozenset((a, b))
                if key == e_remove:
                    continue
                if key in edges_set:
                    continue
                candidates.append((a, b))
            
            if not candidates:
                temp *= cooling
                continue
            
            # Pick a candidate. We can bias towards edges between low-degree nodes
            # or just pick randomly from candidates
            # To speed up, let's just pick randomly but evaluate
            a, b = random.choice(candidates)
            
            new_edges_set = set(edges_set)
            new_edges_set.remove(e_remove)
            new_edges_set.add(frozenset((a, b)))
            
            if len(new_edges_set) != target_edges:
                temp *= cooling
                continue
            
            new_adj = _build_adjacency(n, [list(e) for e in new_edges_set])
            new_score = _max_triangle_degree_sum(n, new_adj)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_score = new_score
                edges = [list(e) for e in new_edges_set]
                adj = new_adj
                if current_score < best_score:
                    best_score = current_score
                    best_edges = edges[:]
            
            temp *= cooling
    
    return {'n': n, 'edges': best_edges}

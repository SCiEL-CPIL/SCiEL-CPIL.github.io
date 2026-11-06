import time
import random
import numpy as np
from itertools import combinations

def construct():
    # Define the two instances
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    total_time_limit = 140.0  # Leave 10s safety margin for the evaluator
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        remaining_time = total_time_limit - elapsed
        
        # If we don't have enough time, just do the trivial solution
        if remaining_time < 1.0:
            cliques = trivial_partition(n, p, seed)
        else:
            # Allocate time proportionally to problem size, but with a minimum
            # Give at least 5s to each instance
            if idx == 0:
                instance_time_limit = min(remaining_time * 0.4, 40.0)
            else:
                instance_time_limit = min(remaining_time * 0.6, 100.0)
            
            cliques = solve_instance(n, p, seed, instance_time_limit)
            
        results.append(cliques)
        
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def generate_edges(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def solve_instance(n, p, seed, time_limit):
    edges = generate_edges(n, p, seed)
    if not edges:
        return []
        
    # Build adjacency list for residual graph
    # res_adj[u] is a set of neighbors v such that (u,v) is in residual graph
    res_adj = {i: set() for i in range(n)}
    for u, v in edges:
        res_adj[u].add(v)
        res_adj[v].add(u)
        
    cliques = []
    start_time = time.time()
    
    # To speed up finding max degree, we can just iterate. n is small.
    while any(len(s) > 0 for s in res_adj.values()):
        current_time = time.time()
        if current_time - start_time > time_limit:
            # Timeout: fall back to trivial for remaining edges
            # Collect all remaining edges
            for u in range(n):
                for v in res_adj[u]:
                    if u < v:
                        cliques.append([u, v])
            break
        
        # Find a large clique
        clique = find_large_clique(res_adj, n)
        
        if not clique:
            # Should not happen if edges exist, but safety check
            # Find any edge
            found = False
            for u in range(n):
                for v in res_adj[u]:
                    if u < v:
                        clique = [u, v]
                        found = True
                        break
                if found: break
            if not found:
                break
                
        # Remove edges in clique
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                res_adj[u].discard(v)
                res_adj[v].discard(u)
                
        cliques.append(clique)
        
        # Local improvement: try to extend the clique by adding vertices
        # that connect to all current clique members and have high degree
        if len(clique) < n:
            # Find candidates that connect to all clique members
            candidates = []
            for v in range(n):
                if v in clique:
                    continue
                # Check if v connects to all clique members
                if all(c in res_adj[v] for c in clique):
                    candidates.append(v)
            
            # Try adding the highest-degree candidate
            if candidates:
                best_candidate = max(candidates, key=lambda x: len(res_adj[x]))
                # Verify it's a valid extension
                if all(best_candidate in res_adj[c] for c in clique):
                    # Add it to the clique
                    clique.append(best_candidate)
                    # Remove its edges
                    for c in clique[:-1]:
                        res_adj[best_candidate].discard(c)
                        res_adj[c].discard(best_candidate)
        
    return cliques

def find_large_clique(res_adj, n):
    """Find a large clique using multiple greedy strategies."""
    # Calculate degrees
    degrees = {u: len(res_adj[u]) for u in range(n)}
    
    # Get non-isolated vertices
    active_vertices = [u for u in range(n) if degrees[u] > 0]
    if not active_vertices:
        return []
    
    best_clique = []
    
    # Strategy 1: Greedy from top-degree vertices with multiple orderings
    sorted_by_degree = sorted(active_vertices, key=lambda x: degrees[x], reverse=True)
    
    # Try different starting points
    K = min(30, len(active_vertices))
    for start_v in sorted_by_degree[:K]:
        # Greedy expansion with max-degree-in-candidate-set heuristic
        C = [start_v]
        P = set(res_adj[start_v])
        
        while P:
            # Pick u in P that maximizes: (degree in P, total degree)
            best_u = None
            best_key = (-1, -1)
            for u in P:
                deg_in_P = len(P & res_adj[u])
                key = (deg_in_P, degrees[u])
                if key > best_key:
                    best_key = key
                    best_u = u
            if best_u is None:
                break
            C.append(best_u)
            P = P & res_adj[best_u]
        
        if len(C) > len(best_clique):
            best_clique = C
    
    # Strategy 2: Try greedy with different vertex orderings (random restarts)
    import random as _rnd
    rng = _rnd.Random(42)
    for _ in range(10):
        # Shuffle active vertices and try first few
        shuffled = active_vertices[:]
        rng.shuffle(shuffled)
        for start_v in shuffled[:5]:
            C = [start_v]
            P = set(res_adj[start_v])
            
            while P:
                best_u = None
                best_key = (-1, -1)
                for u in P:
                    deg_in_P = len(P & res_adj[u])
                    key = (deg_in_P, degrees[u])
                    if key > best_key:
                        best_key = key
                        best_u = u
                if best_u is None:
                    break
                C.append(best_u)
                P = P & res_adj[best_u]
            
            if len(C) > len(best_clique):
                best_clique = C
    
    return best_clique

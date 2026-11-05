"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def _random_graph(n, p, seed):
    """Generate random graph edges matching the verifier's RNG sequence."""
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _build_adjacency(n, edges):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _max_clique_greedy(adj, remaining_edges, n):
    """
    Find a large clique greedily. Returns (clique_vertices, edges_covered).
    Uses a heuristic: pick a vertex with high degree in remaining graph,
    then grow a clique by intersecting neighborhoods.
    """
    # Build remaining edge set as adjacency
    rem_adj = [set() for _ in range(n)]
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    # Vertex degrees in remaining graph
    deg = [len(rem_adj[i]) for i in range(n)]
    
    if not remaining_edges:
        return [], []
    
    # Start with the vertex of highest degree
    start = max(range(n), key=lambda i: deg[i])
    if deg[start] == 0:
        return [], []
    
    # Greedy clique building
    clique = [start]
    candidates = set(rem_adj[start])
    
    while candidates:
        # Pick candidate with highest degree within current candidates
        best = max(candidates, key=lambda v: len(rem_adj[v] & candidates))
        # Check if best connects to all current clique members
        if all(best in rem_adj[c] for c in clique):
            clique.append(best)
            candidates &= rem_adj[best]
        else:
            # Try next best
            candidates.discard(best)
            if not candidates:
                break
    
    # Ensure clique size >= 2
    if len(clique) < 2:
        # Fall back to an edge
        u, v = remaining_edges[0]
        clique = [u, v]
    
    # Compute edges covered by this clique
    clique_set = set(clique)
    covered = set()
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            u, v = clique[i], clique[j]
            if (u, v) in rem_adj[u]:  # edge exists
                covered.add((min(u,v), max(u,v)))
            elif (v, u) in rem_adj[u]:
                covered.add((min(u,v), max(u,v)))
    
    # Verify all pairs in clique are actually edges
    valid_clique = []
    for v in clique:
        if all(w in rem_adj[v] for w in valid_clique):
            valid_clique.append(v)
    
    if len(valid_clique) < 2:
        u, v = remaining_edges[0]
        valid_clique = [u, v]
    
    clique_set = set(valid_clique)
    covered = set()
    for i in range(len(valid_clique)):
        for j in range(i + 1, len(valid_clique)):
            u, v = valid_clique[i], valid_clique[j]
            covered.add((min(u,v), max(u,v)))
    
    return valid_clique, covered

def _solve_instance(n, p, seed, time_budget):
    """Solve one instance: partition all edges into minimum number of cliques."""
    start_time = time.time()
    edges = _random_graph(n, p, seed)
    adj = _build_adjacency(n, edges)
    
    if not edges:
        return []
    
    # Represent edges as frozensets for fast lookup
    edge_set = set(frozenset(e) for e in edges)
    remaining = set(edge_set)
    
    cliques = []
    
    while remaining:
        if time.time() - start_time > time_budget * 0.9:
            # Emergency: cover remaining edges individually
            for e in remaining:
                cliques.append(list(e))
            break
        
        # Try to find a good clique
        # Strategy: for each vertex, try to build a clique centered around it
        # Pick the vertex involved in the most remaining edges
        vertex_deg = defaultdict(int)
        for e in remaining:
            for v in e:
                vertex_deg[v] += 1
        
        if not vertex_deg:
            break
        
        # Try several starting vertices (top candidates)
        sorted_vertices = sorted(vertex_deg.keys(), key=lambda v: -vertex_deg[v])
        top_k = min(20, len(sorted_vertices))
        
        best_clique = None
        best_covered = set()
        
        for start_v in sorted_vertices[:top_k]:
            # Build clique starting from start_v
            clique = [start_v]
            candidates = set(adj[start_v]) & set(remaining and [v for e in remaining for v in e if v != start_v])
            
            # More efficient: get neighbors of start_v that are in remaining edges
            start_neighbors = set()
            for e in remaining:
                for v in e:
                    if v == start_v:
                        for other in e:
                            if other != start_v:
                                start_neighbors.add(other)
            
            candidates = start_neighbors & adj[start_v]
            
            while candidates:
                # Pick the candidate with most connections to other candidates
                best_cand = max(candidates, key=lambda v: sum(1 for w in candidates if w != v and w in adj[v]))
                if all(best_cand in adj[c] for c in clique):
                    clique.append(best_cand)
                    new_cands = set()
                    for w in candidates:
                        if w != best_cand and all(w in adj[c] for c in clique):
                            new_cands.add(w)
                    candidates = new_cands
                else:
                    candidates.discard(best_cand)
            
            if len(clique) >= 2:
                # Compute covered edges
                covered = set()
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, v = clique[i], clique[j]
                        ef = frozenset((u, v))
                        if ef in remaining:
                            covered.add(ef)
                
                if len(covered) > len(best_covered):
                    best_clique = clique
                    best_covered = covered
        
        if best_clique is None or not best_covered:
            # Fallback: take one edge
            e = next(iter(remaining))
            best_clique = list(e)
            best_covered = {e}
        
        cliques.append(best_clique)
        remaining -= best_covered
    
    return cliques

def construct():
    start_time = time.time()
    total_time_budget = 120.0
    safety_margin = 10.0
    available_time = total_time_budget - safety_margin
    
    # Split time between two instances
    time_per_instance = available_time / 2.0
    
    result1 = _solve_instance(32, 0.6, 601, time_per_instance)
    result2 = _solve_instance(64, 0.55, 602, time_per_instance)
    
    return [result1, result2]
# EVOLVE-BLOCK-END

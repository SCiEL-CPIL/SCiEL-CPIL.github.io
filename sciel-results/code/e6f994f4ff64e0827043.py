"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _generate_graph(n, p, seed):
    """Generate random graph, return adjacency as set of frozensets and edge list."""
    rng = Random(seed)
    edges = set()
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
    return edges, adj

def _find_greedy_clique(rem_adj, start_vertex):
    """Grow a clique greedily from start_vertex, preferring high-degree candidates."""
    clique = [start_vertex]
    candidates = set(rem_adj[start_vertex])
    while candidates:
        # Pick candidate with max degree within current candidate set
        best = None
        best_score = -1
        for c in candidates:
            # Count how many other candidates are adjacent to c
            common = len(candidates & rem_adj[c])
            if common > best_score:
                best_score = common
                best = c
        if best is None:
            break
        clique.append(best)
        candidates &= rem_adj[best]
    return clique

def _solve_instance(n, p, seed, time_limit=110.0):
    """Solve one instance: partition all edges into minimum cliques."""
    deadline = time.time() + time_limit
    edges, adj = _generate_graph(n, p, seed)
    if not edges:
        return []
    
    remaining_edges = set(edges)
    rem_adj = [set(s) for s in adj]
    cliques = []
    
    while remaining_edges:
        if time.time() > deadline:
            # Fallback: pack remaining edges one by one
            for e in remaining_edges:
                cliques.append([e[0], e[1]])
            break
        
        # Find a good starting vertex: one with most remaining edges
        vertex_edge_count = {}
        for u, v in remaining_edges:
            vertex_edge_count[u] = vertex_edge_count.get(u, 0) + 1
            vertex_edge_count[v] = vertex_edge_count.get(v, 0) + 1
        
        if not vertex_edge_count:
            break
        
        # Try multiple starting vertices and pick the one that yields the largest clique
        best_clique = None
        best_clique_size = 0
        
        # Sort vertices by edge count descending, try top few
        sorted_vertices = sorted(vertex_edge_count.items(), key=lambda x: -x[1])
        vertices_to_try = [v for v, _ in sorted_vertices[:min(20, len(sorted_vertices))]]
        
        for start in vertices_to_try:
            if time.time() > deadline:
                break
            clique = _find_greedy_clique(rem_adj, start)
            if len(clique) > best_clique_size:
                best_clique_size = len(clique)
                best_clique = clique
        
        if best_clique is None or len(best_clique) < 2:
            # No clique found; just take one edge
            e = next(iter(remaining_edges))
            cliques.append([e[0], e[1]])
            remaining_edges.discard(e)
            continue
        
        # Extract all edges within the clique
        clique_set = set(best_clique)
        clique_edges = set()
        for i, u in enumerate(best_clique):
            for v in best_clique[i+1:]:
                if (u, v) in remaining_edges:
                    clique_edges.add((u, v))
                elif (v, u) in remaining_edges:
                    clique_edges.add((v, u))
        
        if not clique_edges:
            # Shouldn't happen, but safety
            e = next(iter(remaining_edges))
            cliques.append([e[0], e[1]])
            remaining_edges.discard(e)
            continue
        
        cliques.append(sorted(best_clique))
        remaining_edges -= clique_edges
        # Update rem_adj to remove the edges just covered
        for u, v in clique_edges:
            rem_adj[u].discard(v)
            rem_adj[v].discard(u)
    
    return cliques

def construct():
    total_deadline = 115.0
    start_time = time.time()
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        elapsed = time.time() - start_time
        remaining_time = total_deadline - elapsed - 2.0  # safety margin
        if remaining_time < 5.0:
            # Not enough time; do simple fallback for this instance
            edges, _ = _generate_graph(n, p, seed)
            results.append([list(e) for e in edges])
        else:
            result = _solve_instance(n, p, seed, time_limit=remaining_time)
            results.append(result)
    
    return results
# EVOLVE-BLOCK-END

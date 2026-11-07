"""Improved program using a greedy maximal clique partition heuristic."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    from random import Random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_cliques(n, k, edges):
        # Build adjacency sets for fast lookup
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Track which edges are already covered
        edge_covered = set()
        for u, v in edges:
            edge_covered.add((min(u, v), max(u, v)))
        
        remaining_edges = set(edge_covered)
        cliques = []
        
        # Precompute: for each vertex, the set of neighbors
        # We'll use a greedy approach: repeatedly find a large clique
        # among the remaining edges' vertices
        
        while remaining_edges:
            # Get all vertices incident to remaining edges
            incident_vertices = set()
            for u, v in remaining_edges:
                incident_vertices.add(u)
                incident_vertices.add(v)
            
            # Start with the vertex that has the most remaining edges
            vertex_edge_count = defaultdict(int)
            for u, v in remaining_edges:
                vertex_edge_count[u] += 1
                vertex_edge_count[v] += 1
            
            if not vertex_edge_count:
                break
                
            # Start clique from the vertex with highest degree in remaining graph
            start_vertex = max(vertex_edge_count.keys(), key=lambda x: vertex_edge_count[x])
            clique = [start_vertex]
            
            # Try to add vertices one by one to form a maximal clique
            # A vertex can be added if it's adjacent to all current clique members
            # and shares at least one remaining edge with the clique
            candidates = list(incident_vertices - {start_vertex})
            candidates.sort(key=lambda x: -vertex_edge_count[x])
            
            for cand in candidates:
                if cand in clique:
                    continue
                # Check if cand is adjacent to all vertices in current clique
                is_clique = all(cand in adj[v] for v in clique)
                if not is_clique:
                    continue
                # Check if adding cand covers at least one new remaining edge
                covers_new = False
                for v in clique:
                    e = (min(cand, v), max(cand, v))
                    if e in remaining_edges:
                        covers_new = True
                        break
                if covers_new:
                    clique.append(cand)
            
            # If we have a valid clique (at least 2 vertices), add it
            if len(clique) >= 2:
                cliques.append(clique)
                # Remove all edges covered by this clique
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        if e in remaining_edges:
                            remaining_edges.discard(e)
            else:
                # Fallback: take any remaining edge as a 2-vertex clique
                u, v = next(iter(remaining_edges))
                cliques.append([u, v])
                remaining_edges.discard((min(u, v), max(u, v)))
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_cliques(n, k, edges)
        # Convert all vertices to Python int
        cliques = [[int(v) for v in clique] for clique in cliques]
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END

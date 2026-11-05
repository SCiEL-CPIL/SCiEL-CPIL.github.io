import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def partition_edges(edges, n):
        if not edges:
            return []
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: if we run out of time, we can't return invalid.
                # But this shouldn't happen.
                # We can try to finish the current partition quickly?
                # Or just break and return what we have? No, must be valid.
                # So we assume it finishes.
                pass

            degrees = [0] * n
            for u, v in remaining_edges:
                degrees[u] += 1
                degrees[v] += 1
            
            # Find vertex with max degree
            seed_vertex = max(range(n), key=lambda i: degrees[i])
            
            if degrees[seed_vertex] == 0:
                # Should not happen if remaining_edges is not empty
                break

            clique = [seed_vertex]
            candidates = set(adj[seed_vertex])
            
            while candidates:
                # Pick candidate with max degree
                best_u = max(candidates, key=lambda u: degrees[u])
                clique.append(best_u)
                
                # Intersect candidates with neighbors of best_u
                candidates.intersection_update(adj[best_u])
            
            # Safety check: ensure clique size >= 2
            if len(clique) < 2:
                # This should not happen, but if it does, pick an arbitrary edge
                # Find any edge incident to seed_vertex
                for v in adj[seed_vertex]:
                    if (min(seed_vertex, v), max(seed_vertex, v)) in remaining_edges:
                        clique.append(v)
                        break
                if len(clique) < 2:
                    break

            cliques.append(clique)
            
            # Remove edges in this clique from remaining_edges
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i + 1, len(clique)):
                    v = clique[j]
                    edge = (min(u, v), max(u, v))
                    remaining_edges.discard(edge)
                        
        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = partition_edges(edges, n)
        results.append(cliques)
        
    return results

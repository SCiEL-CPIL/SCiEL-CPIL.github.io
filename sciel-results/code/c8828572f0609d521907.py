"""New program using Heuristic Maximum Clique Partitioning."""
import random

def construct():
    def instance(n, p, seed):
        r = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if r.random() < p:
                    edges.append((u, v))
        return edges

    def heuristic_clique_partition(edges, n):
        # Build adjacency as bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        
        remaining_edges = set(edges)
        cliques = []

        while remaining_edges:
            # Calculate degrees in the remaining graph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Pick start vertex with max degree
            start_v = max(range(n), key=lambda x: deg[x])
            
            # Greedy expansion to form a large clique
            clique = [start_v]
            # Candidates must be neighbors of all vertices in clique
            # Start with neighbors of start_v
            candidates_mask = adj[start_v]
            
            while True:
                # Find best candidate
                best_cand = -1
                best_score = -1
                
                # Iterate through bits in candidates_mask
                m = candidates_mask
                while m:
                    # Get lowest set bit
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    
                    # Check if v connects to all in clique via remaining edges
                    valid = True
                    for c in clique:
                        if (min(v, c), max(v, c)) not in remaining_edges:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: degree of v in the remaining graph
                    score = deg[v]
                    
                    if score > best_score:
                        best_score = score
                        best_cand = v
                
                if best_cand == -1:
                    break
                
                clique.append(best_cand)
                
                # Update candidates: must be neighbor of best_cand
                candidates_mask &= adj[best_cand]
                
                if candidates_mask == 0:
                    break

            # We found a clique 'clique'
            if len(clique) >= 2:
                cliques.append(clique)
                # Remove all edges within this clique from remaining_edges
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, v = clique[i], clique[j]
                        remaining_edges.discard((min(u, v), max(u, v)))
            else:
                # Fallback: take one edge
                u, v = next(iter(remaining_edges))
                cliques.append([u, v])
                remaining_edges.discard((u, v))

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = instance(n, p, seed)
        cliques = heuristic_clique_partition(edges, n)
        result.append(cliques)

    return result

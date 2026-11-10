"""Clique partition of split graphs using specialized greedy set cover on cross-edges."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build edge set
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        # Separate clique-part edges and cross-edges
        clique_edges = set()
        cross_edges = set()
        
        for (u, v) in edge_set:
            if u < k and v < k:
                clique_edges.add((u, v))
            elif u >= k and v >= k:
                # Should not exist in split graph
                pass
            else:
                # Cross edge
                a, b = min(u, v), max(u, v)
                cross_edges.add((a, b))
        
        # The clique-part vertices 0..k-1 form a complete graph
        # We can cover all clique edges with one clique: [0, 1, ..., k-1]
        cliques = []
        if k >= 2:
            cliques.append(list(range(k)))
        
        # Now handle cross-edges using greedy set cover
        # Each cross edge is (c, i) where c < k (clique vertex) and i >= k (independent vertex)
        # A clique of the form {c} U S where S is a subset of independent vertices all connected to c
        # But also cliques can be {c1, c2, i} if i is connected to both c1 and c2
        # For simplicity and efficiency, we use cliques centered at independent vertices:
        # For each independent vertex i, its neighbors in clique are N(i)
        # We can form clique {i} U S where S is a subset of N(i)
        # This clique covers all cross edges (c, i) for c in S
        
        # Build adjacency for independent vertices to clique vertices
        # adj[i] = set of clique vertices connected to i
        adj = [set() for _ in range(n)]
        for (u, v) in cross_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each independent vertex i (i >= k), adj[i] contains its clique neighbors
        # We want to cover all cross edges with cliques of form {i} U S where S subset of adj[i]
        # Greedy: repeatedly pick the clique that covers the most uncovered cross edges
        
        remaining_cross = set(cross_edges)
        
        # Precompute for each independent vertex i, the set of cross edges incident to i
        # cross_edges_by_i[i] = set of (c, i) for c in adj[i]
        cross_edges_by_i = {}
        for i in range(k, n):
            cross_edges_by_i[i] = set()
        
        for (u, v) in cross_edges:
            # u < k, v >= k or u >= k, v < k
            if u < k:
                c, i = u, v
            else:
                c, i = v, u
            cross_edges_by_i[i].add((c, i))
        
        while remaining_cross:
            if time.time() > deadline - 5:
                # Fallback: list each remaining cross edge as a 2-clique
                for (c, i) in remaining_cross:
                    cliques.append([int(c), int(i)])
                break
            
            # Greedy set cover: find the clique (centered at some independent vertex i)
            # that covers the most uncovered cross edges
            # For each independent vertex i, the best clique is {i} U (all uncovered neighbors in adj[i])
            # This covers all uncovered cross edges incident to i
            
            best_i = -1
            best_count = 0
            best_clique = None
            
            for i in range(k, n):
                if not cross_edges_by_i[i]:
                    continue
                # Count how many cross edges incident to i are still uncovered
                count = 0
                for (c, j) in cross_edges_by_i[i]:
                    if (c, j) in remaining_cross:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_i = i
                    # The clique is {i} U {c : (c, i) in remaining_cross}
                    S = set()
                    for (c, j) in cross_edges_by_i[i]:
                        if (c, j) in remaining_cross:
                            S.add(c)
                    if S:
                        best_clique = [i] + sorted(S)
            
            if best_i == -1 or best_count == 0:
                # Should not happen if remaining_cross is non-empty
                # Fallback: take any edge
                c, i = next(iter(remaining_cross))
                cliques.append([int(c), int(i)])
                remaining_cross.discard((c, i))
                continue
            
            cliques.append(best_clique)
            
            # Remove all cross edges covered by this clique
            clique_set = set(best_clique)
            new_remaining = set()
            for (u, v) in remaining_cross:
                if u in clique_set and v in clique_set:
                    continue
                new_remaining.add((u, v))
            remaining_cross = new_remaining
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        return final_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def popcount(x):
        return bin(x).count("1")

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # 1. Compute bitmask for each v in I
        # K vertices are 0..k-1
        masks = {}
        for v in range(k, n):
            m = 0
            for u in neighbors[v]:
                m |= (1 << u)
            masks[v] = m
            
        # 2. Greedy selection of S
        # Sort vertices in I by size of N(v) descending
        sorted_v = sorted(range(k, n), key=lambda v: popcount(masks[v]), reverse=True)
        
        S = []
        # For fast conflict check, we store the masks of selected vertices
        selected_masks = []
        
        for v in sorted_v:
            m = masks[v]
            conflict = False
            for sm in selected_masks:
                # Check if intersection has size > 1
                # Intersection size > 1 means there is a pair {u,w} in both
                # which means the edge (u,w) is covered twice
                if popcount(m & sm) > 1:
                    conflict = True
                    break
            if not conflict:
                S.append(v)
                selected_masks.append(m)
        
        cliques = []
        
        # 3. Create cliques for S
        # And 2-cliques for I \ S
        covered_k_edges = set()
        
        for v in S:
            # Clique is N(v) U {v}
            clique = list(neighbors[v]) + [v]
            cliques.append(clique)
            # Mark K-edges within N(v) as covered
            n_v = neighbors[v]
            for i in range(len(n_v)):
                for j in range(i + 1, len(n_v)):
                    covered_k_edges.add((n_v[i], n_v[j]))
        
        for v in range(k, n):
            if v in S:
                continue
            # Use 2-cliques for cross edges
            for u in neighbors[v]:
                cliques.append([u, v])
                
        # 4. Cover remaining K-edges
        # Total K-edges: all pairs in 0..k-1
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_k_edges:
                    remaining_edges.add((u, w))
        
        # Greedy clique cover for remaining_edges
        # Graph on K vertices, edges are remaining_edges
        # We represent adjacency for the subgraph of remaining edges
        # adj[u] is a set of v such that (u,v) in remaining_edges
        adj = {u: set() for u in range(k)}
        for u, w in remaining_edges:
            adj[u].add(w)
            adj[w].add(u)
            
        while remaining_edges:
            # Pick an arbitrary edge to start a clique
            # Heuristic: pick edge with endpoints having high degree?
            # For simplicity, just pick the first one
            u, w = next(iter(remaining_edges))
            
            # Build maximal clique containing u and w in the graph (V, remaining_edges)
            clique = {u, w}
            # Candidates must be neighbors of u AND neighbors of w in the current graph
            candidates = adj[u] & adj[w]
            
            # Iterate through candidates and add if connected to all in clique
            # Since we are adding to a clique, we just need to check connection to u and w initially,
            # and then to any newly added vertices.
            # Actually, since we start with {u,w}, and candidates are neighbors of both,
            # any candidate is connected to u and w.
            # If we add x, we must ensure x is connected to all other added vertices.
            
            added = []
            for c in candidates:
                # Check if c is connected to all in 'added'
                ok = True
                for a in added:
                    if a not in adj[c]:
                        ok = False
                        break
                if ok:
                    clique.add(c)
                    added.append(c)
            
            # Add this clique to the solution
            cliques.append(list(clique))
            
            # Remove all edges within this clique from remaining_edges and adj
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    e1 = (clique_list[i], clique_list[j])
                    e2 = (clique_list[j], clique_list[i])
                    if e1 in remaining_edges:
                        remaining_edges.remove(e1)
                    if e2 in remaining_edges:
                        remaining_edges.remove(e2)
                    # Update adj
                    if clique_list[j] in adj[clique_list[i]]:
                        adj[clique_list[i]].remove(clique_list[j])
                    if clique_list[i] in adj[clique_list[j]]:
                        adj[clique_list[j]].remove(clique_list[i])
                        
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            # Fallback to trivial if time runs out (unlikely)
            # But we should have finished.
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    return results

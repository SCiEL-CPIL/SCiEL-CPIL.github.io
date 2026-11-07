def construct():
    from random import Random
    import itertools

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Vertices 0..k-1 are C, k..n-1 are I
        # Edges in C
        edges_c = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges_c.add((u, v))
        
        # Edges between C and I
        edges_ci = set()
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_ci.add((u, v))
        
        # Build adjacency for I to C
        # N(v) for v in I
        neighbors_in_c = {v: set() for v in range(k, n)}
        for (u, v) in edges_ci:
            neighbors_in_c[v].add(u)
            # Note: edges are stored as (u, v) with u < v? 
            # In my loop u < k <= v, so u < v.
            # But edges_c has u < v.
            
        # Step 1: Form cliques for each v in I with neighbors
        cliques = []
        covered_edges = set()
        
        for v in range(k, n):
            Nv = neighbors_in_c[v]
            if len(Nv) > 0:
                # Clique is {v} U Nv
                clique_vertices = list(Nv) + [v]
                cliques.append(clique_vertices)
                
                # Mark edges covered by this clique
                # Edges within Nv
                Nv_list = list(Nv)
                for i in range(len(Nv_list)):
                    for j in range(i + 1, len(Nv_list)):
                        u1, u2 = Nv_list[i], Nv_list[j]
                        edge = (min(u1, u2), max(u1, u2))
                        covered_edges.add(edge)
                # Edges between v and Nv
                for u in Nv:
                    edge = (min(u, v), max(u, v))
                    covered_edges.add(edge)
        
        # Step 2: Find remaining edges in C
        remaining_edges = set()
        for e in edges_c:
            if e not in covered_edges:
                remaining_edges.add(e)
        
        # Step 3: Partition remaining_edges into minimum cliques
        # Build graph G_rem on C with edges remaining_edges
        # We need to find min number of cliques to cover these edges.
        # Greedy: repeatedly find a maximal clique in the current graph of uncovered edges.
        
        current_edges = set(remaining_edges)
        
        while current_edges:
            # Find a maximal clique in the graph defined by current_edges
            # Since k is small, we can do this simply.
            # Pick an edge (u, v). Try to extend it to a maximal clique.
            # To be more efficient, we can just pick the first edge and find the maximal clique containing it.
            # Or, find the densest clique?
            # Given sparsity, any maximal clique is likely small.
            
            # Heuristic: Pick the edge with the highest "clique potential"?
            # Or just pick the first one.
            u, v = next(iter(current_edges))
            
            # Find maximal clique containing u and v in G_current
            # Start with S = {u, v}
            # Add any vertex w such that (u, w) and (v, w) are in current_edges
            S = {u, v}
            candidates = set()
            for edge in current_edges:
                a, b = edge
                if a in S and b not in S:
                    candidates.add(b)
                elif b in S and a not in S:
                    candidates.add(a)
            
            # Greedily add candidates if they connect to all in S
            # Actually, simpler:
            # A vertex w can be added to S if it is connected to all vertices in S in G_current.
            changed = True
            while changed:
                changed = False
                for w in list(candidates):
                    if w in S:
                        continue
                    # Check if w is connected to all in S
                    is_connected = True
                    for x in S:
                        e = (min(w, x), max(w, x))
                        if e not in current_edges:
                            is_connected = False
                            break
                    if is_connected:
                        S.add(w)
                        changed = True
                        # Update candidates? Not strictly necessary if we just check all remaining vertices
                        # But removing from candidates helps.
                        candidates.remove(w)
            
            # Now S is a clique in G_current.
            # Remove all edges within S from current_edges
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    e = (min(a, b), max(a, b))
                    if e in current_edges:
                        current_edges.remove(e)
            
            # Add this clique to the partition
            cliques.append(S_list)
        
        # Return the list of cliques
        # Ensure types are correct: list of lists of ints
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        results.append(instance(*spec))
    return results

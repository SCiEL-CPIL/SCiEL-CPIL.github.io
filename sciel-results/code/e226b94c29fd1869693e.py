def construct():
    import time
    from random import Random

    deadline = time.time() + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed):
        cross_edges = instance(n, k, p, seed)
        
        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)
            
        # Available C-edges
        # Using a set of tuples (min, max) for O(1) lookup
        available = set()
        for u in range(k):
            for w in range(u + 1, k):
                available.add((u, w))
                
        mixed_cliques = []
        
        # Sort I vertices by degree descending to handle "harder" ones first?
        # Or just process in order.
        # Let's try processing in order of decreasing |N[v]|
        I_vertices = sorted(range(k, n), key=lambda x: len(N[x]), reverse=True)
        
        for v in I_vertices:
            if not N[v]:
                continue
                
            remaining_neighbors = set(N[v])
            
            while remaining_neighbors:
                if time.time() > deadline:
                    # Fallback: just add 2-cliques for remaining
                    for u in remaining_neighbors:
                        mixed_cliques.append([int(v), int(u)])
                    break
                
                # Find a subset S of remaining_neighbors such that all edges in S are in available
                # Greedy: pick a vertex, then add others if compatible
                # To maximize S, we can try to find a maximal clique in the graph induced by remaining_neighbors on 'available' edges.
                
                # Simple greedy:
                # Pick a random or first vertex
                # Try to add more vertices that are connected to all current in S via available edges
                
                # Let's try a simple heuristic:
                # Start with empty S
                # Iterate through remaining_neighbors
                # If u can be added to S (all edges from u to S are in available), add it
                
                # To get a larger S, we might want to sort candidates?
                # Let's just pick the first one and expand.
                
                S = []
                S_set = set()
                
                # Order of addition matters.
                # Let's try adding vertices in a way that keeps options open?
                # Or just greedy.
                
                # A better greedy:
                # Pick a vertex u from remaining_neighbors.
                # S = [u]
                # Then try to add others.
                
                # Let's pick the vertex with highest degree in the 'available' graph restricted to remaining_neighbors?
                # Or just the first one.
                
                # Let's try:
                # 1. Pick arbitrary u from remaining_neighbors.
                # 2. S = [u].
                # 3. For each other w in remaining_neighbors:
                #    If all (min(w,x), max(w,x)) in available for all x in S:
                #       Add w to S.
                
                # To improve, we can restart with different starting vertices?
                # Given time limits, simple greedy might suffice.
                
                # Let's try to find the largest possible S using a simple backtracking if k is small?
                # k is up to 30. |remaining_neighbors| <= 30.
                # Maximal clique in graph of size 30 is fast.
                
                # Let's implement a greedy maximal clique finder.
                
                candidates = list(remaining_neighbors)
                # Sort candidates by degree in available graph within candidates?
                # deg(w) = count of x in candidates s.t. (min(w,x), max(w,x)) in available
                
                def get_deg(w, cand_set):
                    cnt = 0
                    for x in cand_set:
                        if x == w: continue
                        e = (min(w, x), max(w, x))
                        if e in available:
                            cnt += 1
                    return cnt
                    
                # Greedy:
                # Pick w with max deg
                # Add to S
                # Remove w and all non-neighbors from candidates
                # Repeat
                
                S = []
                S_set = set()
                cand_set = set(remaining_neighbors)
                
                while cand_set:
                    # Find vertex with max degree in cand_set
                    best_v = None
                    best_deg = -1
                    for w in cand_set:
                        deg = 0
                        for x in cand_set:
                            if x == w: continue
                            e = (min(w, x), max(w, x))
                            if e in available:
                                deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_v = w
                            
                    if best_v is None:
                        break
                        
                    S.append(best_v)
                    S_set.add(best_v)
                    
                    # Update cand_set: keep only neighbors of best_v in cand_set
                    new_cand = set()
                    for w in cand_set:
                        if w == best_v: continue
                        e = (min(best_v, w), max(best_v, w))
                        if e in available:
                            new_cand.add(w)
                    cand_set = new_cand
                    
                if S:
                    mixed_cliques.append([int(v)] + [int(u) for u in S])
                    # Remove edges in S from available
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            e = (min(S[i], S[j]), max(S[i], S[j]))
                            available.discard(e)
                    # Remove S from remaining_neighbors
                    remaining_neighbors.difference_update(S_set)
                else:
                    # Should not happen if remaining_neighbors not empty and available has edges?
                    # If no edges available between any pair in remaining_neighbors, S will be size 1.
                    # If S is empty, it means we couldn't pick any?
                    # The loop picks best_v. If cand_set is not empty, best_v is picked.
                    # So S will have at least 1.
                    pass

        # Now cover remaining available edges with pure cliques
        pure_cliques = []
        rem_edges = set(available)
        
        while rem_edges:
            if time.time() > deadline:
                # Fallback: add edges as 2-cliques
                for e in rem_edges:
                    pure_cliques.append([int(e[0]), int(e[1])])
                break
                
            # Find a maximal clique in the graph defined by rem_edges
            # Vertices involved in rem_edges
            vertices = set()
            for u, w in rem_edges:
                vertices.add(u)
                vertices.add(w)
                
            # Greedy maximal clique
            S = []
            S_set = set()
            cand_set = set(vertices)
            
            while cand_set:
                # Pick vertex with max degree in cand_set (within rem_edges)
                best_v = None
                best_deg = -1
                for w in cand_set:
                    deg = 0
                    for x in cand_set:
                        if x == w: continue
                        e = (min(w, x), max(w, x))
                        if e in rem_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = w
                            
                if best_v is None:
                    break
                
                S.append(best_v)
                S_set.add(best_v)
                
                # Update cand_set
                new_cand = set()
                for w in cand_set:
                    if w == best_v: continue
                    e = (min(best_v, w), max(best_v, w))
                    if e in rem_edges:
                        new_cand.add(w)
                cand_set = new_cand
                
            if S:
                pure_cliques.append([int(u) for u in S])
                # Remove edges in S from rem_edges
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        rem_edges.discard(e)
            else:
                # If no clique found (shouldn't happen if rem_edges not empty), add 2-clique
                # Pick any edge
                e = next(iter(rem_edges))
                pure_cliques.append([int(e[0]), int(e[1])])
                rem_edges.discard(e)

        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for (n, k, p, seed) in specs:
        if time.time() > deadline:
            # Fallback for safety
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
        else:
            result.append(solve_instance(n, k, p, seed))

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final

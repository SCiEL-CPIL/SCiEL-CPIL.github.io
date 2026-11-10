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

    def solve_instance(n, k, p, seed, variant_seed=None):
        cross_edges = instance(n, k, p, seed)
        
        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)
            
        # Available C-edges
        available = set()
        for u in range(k):
            for w in range(u + 1, k):
                available.add((u, w))
                
        mixed_cliques = []
        
        # Determine order of I vertices
        if variant_seed is None:
            I_vertices = sorted(range(k, n), key=lambda x: len(N[x]), reverse=False)
            rng_var = None
        else:
            rng_var = Random(variant_seed)
            # Strategy 1: Random order
            # Strategy 2: Degree order with noise
            # Let's alternate or use seed to decide
            if variant_seed % 2 == 0:
                I_vertices = list(range(k, n))
                rng_var.shuffle(I_vertices)
            else:
                # Sort by degree, but break ties randomly? Or just sort.
                # To make it different from base, let's sort by (degree + small_noise)
                I_vertices = sorted(range(k, n), key=lambda x: len(N[x]) + rng_var.random() * 0.1, reverse=True)

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
                
                S = []
                S_set = set()
                cand_set = set(remaining_neighbors)
                
                while cand_set:
                    # Find vertex with max degree in cand_set
                    best_v = None
                    best_deg = -1
                    candidates_for_best = []
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
                            candidates_for_best = [w]
                        elif deg == best_deg:
                            candidates_for_best.append(w)
                            
                    if best_v is None:
                        break
                    
                    # Tie-breaking
                    if len(candidates_for_best) > 1 and rng_var is not None:
                        best_v = rng_var.choice(candidates_for_best)
                    
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
                    # Should not happen if remaining_neighbors not empty
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
                
            # Pick the edge with max combined degree
            best_edge = None
            best_score = -1
            for e in rem_edges:
                u, w = e
                deg_u = sum(1 for x in rem_edges if u in x)
                deg_w = sum(1 for x in rem_edges if w in x)
                score = deg_u + deg_w
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                break
            
            # Find max clique containing best_edge
            u0, w0 = best_edge
            cand_set = {x for e in rem_edges for x in e if (x == u0 or x == w0 or (min(u0,x),max(u0,x)) in rem_edges and min(w0,x),max(w0,x) in rem_edges)}
            # Simpler: start with {u0, w0} and greedily add
            S = [u0, w0]
            S_set = {u0, w0}
            cand = {x for e in rem_edges for x in e if x not in S_set}
            
            while cand:
                # Pick vertex with max degree in cand that connects to all in S
                best_v = None
                best_deg = -1
                for w in cand:
                    # Check if w connects to all in S
                    ok = True
                    deg = 0
                    for s in S:
                        e = (min(w, s), max(w, s))
                        if e not in rem_edges:
                            ok = False
                            break
                        deg += 1
                    if ok and deg > best_deg:
                        best_deg = deg
                        best_v = w
                if best_v is None:
                    break
                S.append(best_v)
                S_set.add(best_v)
                # Update cand
                new_cand = set()
                for w in cand:
                    if w == best_v: continue
                    ok = True
                    for s in S:
                        e = (min(w, s), max(w, s))
                        if e not in rem_edges:
                            ok = False
                            break
                    if ok:
                        new_cand.add(w)
                cand = new_cand
            
            if len(S) >= 2:
                pure_cliques.append([int(x) for x in S])
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        rem_edges.discard(e)
            else:
                e = best_edge
                pure_cliques.append([int(e[0]), int(e[1])])
                rem_edges.discard(e)

        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    
    for (n, k, p, seed) in specs:
        best_cliques = None
        best_count = float('inf')
        
        # Run multiple variants
        # 1. Base (None)
        # 2-20. Variants with random seeds
        if time.time() > deadline:
            # Fallback for safety
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
            continue
            
        # Try base first
        cliques = solve_instance(n, k, p, seed, None)
        count = len(cliques)
        if count < best_count:
            best_count = count
            best_cliques = cliques
            
        # Try variants
        # Estimate time per run. If fast, do more.
        # Let's do 60 variants.
        for i in range(60):
            if time.time() > deadline - 10.0: # Safety margin
                break
            v_seed = seed + 1000 + i
            cliques = solve_instance(n, k, p, seed, v_seed)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
                
        result.append(best_cliques)

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final

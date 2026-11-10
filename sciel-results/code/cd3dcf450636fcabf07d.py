def construct():
    from random import Random
    import time
    from itertools import combinations

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # Clique on first k vertices
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate cross edges and C-C edges
        cross_neighbors = {}  # v (in I) -> set of C-neighbors
        cc_edges = set()  # C-C edges
        
        for (a, b) in edges:
            if a < k and b < k:
                cc_edges.add((min(a,b), max(a,b)))
            elif a >= k and b < k:
                if a not in cross_neighbors:
                    cross_neighbors[a] = set()
                cross_neighbors[a].add(b)
            elif a < k and b >= k:
                if b not in cross_neighbors:
                    cross_neighbors[b] = set()
                cross_neighbors[b].add(a)
        
        # Build CC adjacency for quick lookup
        cc_adj = {i: set() for i in range(k)}
        for (a, b) in cc_edges:
            cc_adj[a].add(b)
            cc_adj[b].add(a)
        
        # Track which C-C edges are covered
        covered_cc = set()
        cliques = []
        
        # Process I-vertices in order of increasing neighborhood size
        # This preserves C-edge density for later large-neighborhood vertices
        i_vertices = sorted(cross_neighbors.keys(), key=lambda v: len(cross_neighbors[v]))
        
        for v in i_vertices:
            S = set(cross_neighbors[v])
            if not S:
                continue
            
            # Use matching-based pair packing:
            # 1. Find a maximum matching in the graph where edges are "cheap" C-C edges
            # 2. Each matched pair becomes a 3-clique [v, a, b]
            # 3. Unmatched vertices become 2-cliques [v, c]
            
            # Build the graph of available C-C edges within S
            # An edge (a,b) is available if (a,b) in cc_edges and (a,b) not in covered_cc
            # We want to find a maximum matching in this graph to minimize C-edge consumption
            
            # Greedy matching: sort edges by "cost" (degree sum) and pick greedily
            # Lower cost = less impact on remaining structure
            
            # Compute degree of each vertex in the available subgraph
            avail_deg = {c: 0 for c in S}
            avail_edges = []
            for a in S:
                for b in S:
                    if a < b:
                        pair = (a, b)
                        if pair in cc_edges and pair not in covered_cc:
                            avail_edges.append(pair)
                            avail_deg[a] += 1
                            avail_deg[b] += 1
            
            # Sort edges by combined degree (ascending) - cheaper edges first
            avail_edges.sort(key=lambda e: avail_deg[e[0]] + avail_deg[e[1]])
            
            matched = set()
            for (a, b) in avail_edges:
                if a not in matched and b not in matched:
                    matched.add(a)
                    matched.add(b)
                    cliques.append([v, a, b])
                    covered_cc.add((a, b))
            
            # Singles: unmatched C-neighbors
            for c in S:
                if c not in matched:
                    cliques.append([v, c])
        
        # Cover remaining CC edges with pure-C cliques
        remaining_cc = cc_edges - covered_cc
        if remaining_cc:
            rem_edges = set(remaining_cc)
            vertices_involved = set()
            for (a, b) in rem_edges:
                vertices_involved.add(a)
                vertices_involved.add(b)
            
            while rem_edges:
                # Find a maximal clique in the remaining graph
                deg = {v: 0 for v in vertices_involved}
                for (a, b) in rem_edges:
                    deg[a] = deg.get(a, 0) + 1
                    deg[b] = deg.get(b, 0) + 1
                
                if not deg:
                    break
                    
                start = max(deg, key=lambda x: deg[x])
                clique = [start]
                candidates = [v for v in vertices_involved if v != start]
                for v in sorted(candidates, key=lambda x: -deg[x]):
                    ok = True
                    for c in clique:
                        pair = (min(v, c), max(v, c))
                        if pair not in rem_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(v)
                
                if len(clique) < 2:
                    a, b = next(iter(rem_edges))
                    clique = [a, b]
                
                cliques.append(clique)
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        pair = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                        rem_edges.discard(pair)
        
        # Convert to plain Python ints and lists
        result = []
        for clique in cliques:
            result.append([int(x) for x in clique])
        return result
    
    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = solve(*spec)
        results.append(cliques)
    
    return results

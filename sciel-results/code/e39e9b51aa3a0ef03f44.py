import itertools

def construct():
    n = 16
    # Adjacency matrix as list of bitmasks
    adj = [0] * n
    for u in range(n):
        for v in range(u+1, n):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    def has_edge(u, v):
        return (adj[u] >> v) & 1

    def find_q3():
        """
        Finds a Q3 subgraph.
        Returns a list of 12 edges [(u,v), ...] or None.
        Structure:
        c connected to a, b, d (independent)
        ab connected to a, b
        ad connected to a, d
        bd connected to b, d
        abd connected to ab, ad, bd
        """
        # Iterate over center c
        for c in range(n):
            # Get neighbors of c
            neighbors_c = [v for v in range(n) if v != c and has_edge(c, v)]
            
            # Iterate over independent triples (a, b, d) from neighbors_c
            # We need a, b, d to be pairwise non-adjacent
            for a, b, d in itertools.combinations(neighbors_c, 3):
                if has_edge(a, b) or has_edge(a, d) or has_edge(b, d):
                    continue
                
                # Find ab: common neighbor of a and b, not c, not a,b,d
                # And ab must be distinct from ad, bd, abd later
                # Let's just find candidates for ab, ad, bd
                # ab in N(a) ∩ N(b)
                # ad in N(a) ∩ N(d)
                # bd in N(b) ∩ N(d)
                
                # We need to pick specific vertices for ab, ad, bd, abd
                # ab cannot be c, a, b, d
                # ad cannot be c, a, d, b
                # bd cannot be c, b, d, a
                
                # To avoid permutations, we can just iterate possible ab, ad, bd
                # But that might be slow. Let's collect candidates.
                
                cand_ab = []
                for v in range(n):
                    if v in (c, a, b, d): continue
                    if has_edge(a, v) and has_edge(b, v):
                        cand_ab.append(v)
                        
                cand_ad = []
                for v in range(n):
                    if v in (c, a, d, b): continue
                    if has_edge(a, v) and has_edge(d, v):
                        cand_ad.append(v)
                        
                cand_bd = []
                for v in range(n):
                    if v in (c, b, d, a): continue
                    if has_edge(b, v) and has_edge(d, v):
                        cand_bd.append(v)
                        
                if not cand_ab or not cand_ad or not cand_bd:
                    continue
                
                # Try combinations of ab, ad, bd
                # Note: ab, ad, bd must be distinct from each other?
                # In Q3, ab, ad, bd are distinct vertices.
                for ab in cand_ab:
                    for ad in cand_ad:
                        if ad == ab: continue
                        for bd in cand_bd:
                            if bd == ab or bd == ad: continue
                            
                            # Now find abd: common neighbor of ab, ad, bd
                            # abd must be distinct from c, a, b, d, ab, ad, bd
                            found_abd = False
                            for abd in range(n):
                                if abd in (c, a, b, d, ab, ad, bd): continue
                                if has_edge(ab, abd) and has_edge(ad, abd) and has_edge(bd, abd):
                                    found_abd = True
                                    # Check all 12 edges
                                    edges = [
                                        (c, a), (c, b), (c, d),
                                        (a, ab), (a, ad),
                                        (b, ab), (b, bd),
                                        (d, ad), (d, bd),
                                        (ab, abd), (ad, abd), (bd, abd)
                                    ]
                                    # Verify all edges exist
                                    valid = True
                                    for u, v in edges:
                                        if not has_edge(u, v):
                                            valid = False
                                            break
                                    if valid:
                                        return edges
                            # If found_abd is True but no valid set found, continue
        return None

    # Greedy removal
    # To improve score, we can try to remove the edge that is in the most Q3s.
    # But counting is expensive. Let's stick to a simple heuristic:
    # Remove the edge from the found Q3 that has the highest degree sum.
    
    while True:
        q3_edges = find_q3()
        if q3_edges is None:
            break
        
        best_edge = None
        best_score = -1
        for u, v in q3_edges:
            # Degree in current graph
            deg_u = bin(adj[u]).count('1')
            deg_v = bin(adj[v]).count('1')
            score = deg_u + deg_v
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge:
            u, v = best_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    # Convert back to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if has_edge(u, v):
                edges.append([u, v])
                
    return edges

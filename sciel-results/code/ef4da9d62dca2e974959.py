"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        K = list(range(k))
        I = list(range(k, n))
        
        # Build adjacency for independent vertices to clique vertices
        k_to_bit = {u: i for i, u in enumerate(K)}
        
        neigh_mask = {}
        for v in I:
            mask = 0
            for u in K:
                if (min(u, v), max(u, v)) in edge_set:
                    mask |= (1 << k_to_bit[u])
            neigh_mask[v] = mask
        
        # Group independent vertices by their exact neighborhood mask
        groups = defaultdict(list)
        for v in I:
            groups[neigh_mask[v]].append(v)
        
        # For each distinct non-empty neighborhood mask, create a clique S ∪ G
        # where S is the set of K vertices corresponding to the mask bits,
        # and G is the group of I vertices with that mask.
        
        # We need to ensure no K-K edge is covered twice.
        # Since we use one clique per distinct mask, and each clique contains 
        # the K vertices in its mask, two cliques will share K-K edges if 
        # their masks share at least 2 bits.
        
        # To handle this, we need a more careful approach.
        # Let's use a greedy approach that processes groups and tracks 
        # which K-K edges have been covered.
        
        uncovered = set(edge_set)
        cliques = []
        
        # Precompute K-K edges
        K_edges = set()
        for i in range(k):
            for j in range(i + 1, k):
                K_edges.add((i, j))
        
        # Process groups in order of decreasing group size
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
        
        # Track which K-K edges are already covered
        covered_K_edges = set()
        
        for mask, group in sorted_groups:
            if mask == 0:
                # No cross edges to cover for this group
                continue
            
            # Determine which K vertices are in this mask
            S = [k_to_bit[u] for u in K if (mask >> u) & 1]
            S = [K[u] for u in S]  # Convert back to vertex labels
            
            # Check if we can use this clique without double-covering K-K edges
            # A clique S ∪ G covers all K-K edges within S.
            # We can only use this clique if no K-K edge within S is already covered.
            
            can_use = True
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    e = (min(S[i], S[j]), max(S[i], S[j]))
                    if e in covered_K_edges:
                        can_use = False
                        break
                if not can_use:
                    break
            
            if can_use and len(S) >= 1:
                # Use this clique
                clique = S + list(group)
                if len(clique) >= 2:
                    cliques.append(clique)
                    # Mark K-K edges within S as covered
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            covered_K_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                    # Remove cross edges from uncovered
                    for v in group:
                        for u in S:
                            uncovered.discard((min(u, v), max(u, v)))
        
        # Now cover remaining edges
        # Remaining edges are:
        # 1. Cross edges not covered (from groups with mask=0 or conflicts)
        # 2. K-K edges not covered
        
        # For remaining cross edges, handle each independent vertex individually
        for v in I:
            # Find uncovered cross edges incident to v
            incident = [u for u in K if (min(u, v), max(u, v)) in uncovered]
            if not incident:
                continue
            
            # Create a clique {v} ∪ incident
            # But we need to check K-K edge conflicts
            # For simplicity, use {v} ∪ {u} for each u, or find a subset
            
            # Greedy: add vertices one by one
            clique = [v]
            for u in incident:
                # Check if adding u causes K-K edge conflict
                # u is in K, so we need to check if any K-K edge involving u is already covered
                # and if that edge is within the current clique's K vertices
                k_in_clique = [x for x in clique if x < k]
                can_add = True
                for x in k_in_clique:
                    e = (min(u, x), max(u, x))
                    if e in covered_K_edges:
                        can_add = False
                        break
                if can_add:
                    clique.append(u)
            
            if len(clique) >= 2:
                cliques.append(clique)
                # Mark K-K edges within the K part of this clique
                k_part = [x for x in clique if x < k]
                for i in range(len(k_part)):
                    for j in range(i + 1, len(k_part)):
                        covered_K_edges.add((min(k_part[i], k_part[j]), max(k_part[i], k_part[j])))
                # Remove cross edges
                for u in k_part:
                    uncovered.discard((min(u, v), max(u, v)))
        
        # Cover remaining K-K edges
        remaining_K = [e for e in K_edges if e in uncovered]
        if remaining_K:
            # Greedy: find a large subset of K that covers many remaining edges
            # without conflicting with covered_K_edges
            used_vertices = set()
            for e in sorted(remaining_K, key=lambda x: (-x[0], -x[1])):
                u, v = e
                if u not in used_vertices and v not in used_vertices:
                    # Check if we can form a clique {u, v}
                    # Only need to check if (u,v) is not already covered
                    if (u, v) not in covered_K_edges:
                        cliques.append([u, v])
                        covered_K_edges.add((u, v))
                        used_vertices.add(u)
                        used_vertices.add(v)
                        uncovered.discard((u, v))
        
        # Final cleanup: any remaining edges
        for e in list(uncovered):
            u, v = e
            if len([x for x in cliques if u in x and v in x]) == 0:
                cliques.append([u, v])
                uncovered.discard(e)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        results.append(solve_instance(*spec))
    
    return results
# EVOLVE-BLOCK-END

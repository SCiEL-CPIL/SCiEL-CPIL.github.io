"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    start_time = time.time()
    deadline = 115.0  # 120s limit, keep margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges_cc = []
        for u in range(k):
            for v in range(u + 1, k):
                edges_cc.append((u, v))
        
        edges_ci = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_ci.append((u, v))
                    
        return edges_cc, edges_ci

    def solve_split_graph(n, k, edges_ci):
        # Build shadow for each independent vertex as a bitmask (int) of neighbors in clique part
        shadows = {}
        for v in range(k, n):
            mask = 0
            for u, v_edge in edges_ci:
                if v_edge == v:
                    mask |= (1 << u)
            shadows[v] = mask
        
        # Group independent vertices by their shadow (bitmask)
        # Vertices with the same shadow can be grouped together in one clique
        # But we can also group vertices with nested shadows?
        # Actually, if shadow_A is a superset of shadow_B, then a clique containing 
        # both A and B must use shadow_B as the common shadow.
        # So we should group vertices by their exact shadow first, then consider merging.
        
        # Step 1: Group by exact shadow
        shadow_groups = {}
        for v in range(k, n):
            mask = shadows[v]
            if mask not in shadow_groups:
                shadow_groups[mask] = []
            shadow_groups[mask].append(v)
        
        # Now, we have groups of independent vertices with identical shadows.
        # Each group can form a clique with its shadow.
        # But we can also merge groups if one shadow is a subset of another?
        # No, merging requires the intersection to be the common shadow.
        # If we merge group A (shadow S_A) and group B (shadow S_B), the common shadow is S_A & S_B.
        # This is only beneficial if it covers more edges than handling them separately.
        # But since we are minimizing the number of cliques, merging two groups into one 
        # clique reduces the count by 1, even if the common shadow is smaller.
        # So we should merge as many groups as possible.
        
        # Strategy: 
        # 1. Start with each unique shadow as a separate group.
        # 2. Repeatedly merge two groups if their intersection is non-empty?
        #    Actually, we can merge any two groups. The new shadow is the intersection.
        #    The only constraint is that the resulting clique must be valid (size >= 2).
        #    Since each group has at least one independent vertex, and the shadow is non-empty 
        #    (otherwise the vertex has no edges and wouldn't be in edges_ci), the clique is valid.
        #    Wait, if a vertex has no edges, its shadow is 0. It doesn't contribute to edges_ci.
        #    So we only care about vertices with non-zero shadow.
        
        # Let's filter out vertices with shadow 0 (no cross edges)
        active_shadows = {mask: group for mask, group in shadow_groups.items() if mask != 0}
        
        # Now, we want to merge these groups to minimize the number of cliques.
        # Merging two groups A and B into one group with shadow A & B reduces the clique count by 1.
        # We should merge greedily: always merge the two groups whose intersection is "largest" 
        # or just merge any two? 
        # Actually, merging any two groups reduces the count by 1, so we should merge as many as possible.
        # The only reason not to merge is if the intersection becomes 0, which would make the 
        # clique invalid (size 1 if only independent vertices remain, but we always have at least 
        # one independent vertex per group, and the shadow must be non-empty for the clique to 
        # cover any edges).
        # Wait, if the intersection is 0, the clique would consist only of independent vertices, 
        # which is not a clique (they are independent). So we must ensure the intersection is non-empty.
        
        # So the algorithm is:
        # 1. Start with all non-zero shadow groups.
        # 2. While there are at least two groups:
        #    - Find two groups whose intersection is non-empty.
        #    - Merge them (new shadow = intersection, new group = union of vertices).
        #    - If no such pair exists, stop.
        # 3. Each remaining group forms a clique with its shadow.
        
        # To find pairs with non-empty intersection, we can use a greedy approach:
        # Sort groups by shadow size (popcount) descending.
        # For each group, try to merge it with another group that has a non-empty intersection.
        
        # Let's implement a simple greedy merge:
        # 1. List of (shadow_mask, vertex_list)
        # 2. While len(list) > 1:
        #    - Find i, j such that (shadow_i & shadow_j) != 0
        #    - Merge j into i: shadow_i &= shadow_j, vertex_list_i += vertex_list_j, remove j
        #    - If no such pair, break
        
        # This might be slow if we do O(n^2) checks each time. But n is small (unique shadows <= 60).
        
        groups = [(mask, verts) for mask, verts in active_shadows.items()]
        
        # Sort by popcount descending to prioritize larger shadows
        groups.sort(key=lambda x: bin(x[0]).count('1'), reverse=True)
        
        merged = True
        while merged and len(groups) > 1:
            merged = False
            for i in range(len(groups)):
                if merged:
                    break
                for j in range(i + 1, len(groups)):
                    if (groups[i][0] & groups[j][0]) != 0:
                        # Merge j into i
                        new_mask = groups[i][0] & groups[j][0]
                        new_verts = groups[i][1] + groups[j][1]
                        groups[i] = (new_mask, new_verts)
                        groups.pop(j)
                        merged = True
                        # Re-sort to maintain order
                        groups.sort(key=lambda x: bin(x[0]).count('1'), reverse=True)
                        break
        
        # Now, each group in `groups` forms a clique
        cliques = []
        # Main clique
        cliques.append(list(range(k)))
        
        for mask, verts in groups:
            # Convert mask to list of clique vertices
            clique_verts = [u for u in range(k) if (mask >> u) & 1]
            clique = clique_verts + verts
            # Ensure size >= 2
            if len(clique) >= 2:
                cliques.append(clique)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return seed-like result to ensure validity
            n, k, p, seed = spec
            edges_cc, edges_ci = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in edges_ci:
                cliques.append([u, v])
            results.append(cliques)
            continue
            
        n, k, p, seed = spec
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        cliques = solve_split_graph(n, k, edges_ci)
        
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END

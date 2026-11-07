def construct():
    from random import Random
    from itertools import combinations

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # C-C edges (clique on first k vertices)
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_clique_partition(vertices, edge_set):
        """
        Partition 'vertices' into minimum number of cliques (w.r.t. edge_set)
        using a greedy approach: repeatedly pick a maximal clique.
        Returns a list of lists (each a clique).
        """
        if not vertices:
            return []
        
        remaining = list(vertices)
        cliques = []
        
        while remaining:
            # Find a maximal clique in the remaining set
            # We use a simple greedy: start with the vertex that has the most neighbors
            # in the remaining set, then add compatible neighbors.
            
            # Compute degrees within 'remaining'
            deg = {v: 0 for v in remaining}
            rem_set = set(remaining)
            for u in remaining:
                for w in remaining:
                    if u < w and (u, w) in edge_set:
                        deg[u] += 1
                        deg[w] += 1
            
            # Sort by degree descending
            sorted_rem = sorted(remaining, key=lambda x: -deg[x])
            
            # Start with the highest degree vertex
            if not sorted_rem:
                break
                
            current_clique = [sorted_rem[0]]
            current_set = {sorted_rem[0]}
            
            # Try to add other vertices
            for v in sorted_rem[1:]:
                if v in current_set:
                    continue
                # Check if v is connected to all in current_clique
                can_add = True
                for u in current_clique:
                    e = (u, v) if u < v else (v, u)
                    if e not in edge_set:
                        can_add = False
                        break
                if can_add:
                    current_clique.append(v)
                    current_set.add(v)
            
            cliques.append(current_clique)
            remaining = [v for v in remaining if v not in current_set]
            
        return cliques

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate cross edges and core (C-C) edges
        cross_edges = set()
        core_edges = set()
        for u, v in edges:
            if u < k and v < k:
                core_edges.add((u, v))
            elif u >= k and v >= k:
                # Should not happen in split graph construction, but safe
                pass
            else:
                # One in C, one in I
                cross_edges.add((u, v))
        
        # For each vertex v in I (k to n-1), find its neighbors in C
        # N(v) = {u in C | (u,v) in cross_edges}
        neighbors_in_C = {}
        for v in range(k, n):
            neighbors_in_C[v] = set()
            for u in range(k):
                e = (u, v) if u < v else (v, u)
                if e in cross_edges:
                    neighbors_in_C[v].add(u)
        
        # Strategy:
        # 1. For each v in I, partition N(v) into cliques.
        #    The cliques will be {v} U part.
        #    This covers all cross edges incident to v.
        # 2. The core edges covered by these partitions are the edges within each part.
        #    We must ensure no core edge is covered by more than one v's partition.
        #    If a core edge (u,w) is in N(v1) and N(v2), it would be covered by both
        #    if u,w are in the same part for both v1 and v2.
        #    To avoid this, we can simply assign each core edge to be "covered" by
        #    at most one v. But the partitioning is local.
        
        # Alternative Strategy (Simpler and likely effective):
        # Just greedily cover all edges.
        # But we want to exploit structure.
        
        # Let's try the "Conflict Resolution" approach mentioned in thoughts,
        # but implemented carefully.
        
        # Step 1: Initial partition for each v in I is just [N(v)] if N(v) is a clique?
        # N(v) is a subset of C, and C is a clique, so N(v) is always a clique.
        # So initially, for each v, we have one clique: {v} U N(v).
        # This covers all cross edges for v.
        # It also covers all core edges within N(v).
        
        # Step 2: Identify core edges that are covered by multiple v's.
        # A core edge (u,w) is covered by v if u in N(v) and w in N(v).
        # Let cover_count[(u,w)] = number of v such that u,w in N(v).
        
        # If cover_count[(u,w)] > 1, we have a conflict.
        # We need to modify the partitions so that for all but one v, u and w are NOT in the same part.
        
        # To minimize cliques, we want to keep partitions large.
        # Splitting a part S into S\{w} and {w} increases clique count by 1 for that v.
        
        # Algorithm:
        # 1. Start with P_v = [N(v)] for all v.
        # 2. For each core edge (u,w):
        #    Find all v such that u,w in N(v). Let this set be V_conflict.
        #    If |V_conflict| <= 1, no action needed.
        #    Else, pick one v_keep in V_conflict.
        #    For each v in V_conflict \ {v_keep}:
        #        Find the part S in P_v containing u.
        #        If w is in S, split S: remove w from S, add {w} as a new part (if non-empty).
        #        Actually, we just need u and w to be in different parts.
        #        Moving w to a new part (singleton) works.
        
        # Implementation details:
        # We need to maintain P_v as a list of sets (or lists) of vertices in C.
        
        P = {}
        for v in range(k, n):
            if neighbors_in_C[v]:
                P[v] = [set(neighbors_in_C[v])]
            else:
                P[v] = []
        
        # Map from core edge to list of v's that cover it
        # We only care about edges in core_edges
        # But note: u,w in N(v) implies (u,w) is an edge in C, which is always true.
        # So we just need to check which v's contain both u and w in their parts.
        
        # To make this efficient, we can iterate over core edges.
        # k is at most 30, so core edges are ~435.
        # Number of v's is n-k, at most 60.
        
        # Let's precompute for each v, the set of pairs covered.
        # Actually, we just need to resolve conflicts.
        
        # Iterate over all core edges (u,w) with u < w
        for u in range(k):
            for w in range(u + 1, k):
                # Find v's that contain both u and w in the same part
                v_covering = []
                for v in range(k, n):
                    if v not in P:
                        continue
                    for part in P[v]:
                        if u in part and w in part:
                            v_covering.append(v)
                            break # Only need to know if it's in SOME part
                
                if len(v_covering) <= 1:
                    continue
                
                # Conflict: more than one v covers this edge
                # Pick the first one to keep
                v_keep = v_covering[0]
                
                for v in v_covering[1:]:
                    # Find the part containing u (and w)
                    for i, part in enumerate(P[v]):
                        if u in part and w in part:
                            # Split: remove w from part, add {w}
                            part.remove(w)
                            P[v].append({w})
                            break
        
        # Now, construct the list of cliques
        cliques = []
        
        # Type A: Cliques involving I vertices
        # For each v, for each part in P[v], clique is {v} U part
        # But we must ensure part has at least 1 vertex? 
        # The problem says "at least two vertices" for the clique.
        # If part is empty, {v} is size 1 -> Invalid.
        # If part has 1 vertex, {v, u} is size 2 -> Valid.
        # So we must only add cliques if len(part) >= 1.
        
        # Also, we must ensure that we don't add duplicate cliques or cliques that are subsets?
        # No, just list of cliques.
        
        # Track which core edges are covered by these Type A cliques
        covered_core_edges = set()
        
        for v in range(k, n):
            if v not in P:
                continue
            for part in P[v]:
                if not part:
                    continue
                clique = [v] + list(part)
                # Ensure distinct ints
                clique = list(set(clique)) # Safety, though should be distinct
                if len(clique) < 2:
                    continue
                cliques.append(clique)
                
                # Mark core edges within 'part' as covered
                part_list = list(part)
                for i in range(len(part_list)):
                    for j in range(i + 1, len(part_list)):
                        u, w = part_list[i], part_list[j]
                        e = (u, w) if u < w else (w, u)
                        covered_core_edges.add(e)
        
        # Type B: Cover remaining core edges
        remaining_core_edges = core_edges - covered_core_edges
        
        if remaining_core_edges:
            # Greedy clique partition on the subgraph induced by remaining_core_edges
            # Vertices involved:
            verts_involved = set()
            for u, w in remaining_core_edges:
                verts_involved.add(u)
                verts_involved.add(w)
            
            # We need to partition these vertices into cliques such that
            # every edge in remaining_core_edges is covered.
            # This is equivalent to covering the edges of the graph G' = (verts_involved, remaining_core_edges)
            # with cliques.
            
            # Since C is a clique, any subset of C is a clique.
            # So we can just group the vertices arbitrarily?
            # No, we must cover the EDGES.
            # If we group vertices into sets, the cliques are the sets themselves.
            # An edge (u,w) is covered if u and w are in the same set.
            # So we need to partition verts_involved into sets such that
            # for every (u,w) in remaining_core_edges, u and w are in the same set.
            # This means each connected component of the graph (verts_involved, remaining_core_edges)
            # must be contained within a single set.
            # Since C is a clique, the union of a connected component is a clique.
            # So we can just take the connected components!
            
            # Find connected components
            visited = set()
            components = []
            adj = {v: set() for v in verts_involved}
            for u, w in remaining_core_edges:
                adj[u].add(w)
                adj[w].add(u)
            
            for v in verts_involved:
                if v not in visited:
                    # BFS
                    comp = set()
                    queue = [v]
                    visited.add(v)
                    while queue:
                        curr = queue.pop()
                        comp.add(curr)
                        for nb in adj[curr]:
                            if nb not in visited:
                                visited.add(nb)
                                queue.append(nb)
                    components.append(comp)
            
            for comp in components:
                if len(comp) >= 2:
                    cliques.append(list(comp))
                elif len(comp) == 1:
                    # Single vertex, no edges, shouldn't happen if it came from edges
                    pass
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        
        # Validation step (optional, for safety)
        # 1. All cliques have >= 2 distinct int vertices
        # 2. Every edge is in exactly one clique
        # 3. Types are int
        
        # Let's do a quick check to ensure no edge is missed or double-covered
        # This is expensive for large n, but n is small.
        covered_edges = set()
        for clique in cliques:
            if len(clique) < 2:
                continue
            # Check types
            for v in clique:
                if not isinstance(v, int) or isinstance(v, bool):
                    raise ValueError("Invalid vertex type")
            if len(set(clique)) != len(clique):
                raise ValueError("Duplicate vertices in clique")
            
            for u, w in combinations(clique, 2):
                e = (u, w) if u < w else (w, u)
                if e in covered_edges:
                    raise ValueError("Edge covered twice: " + str(e))
                covered_edges.add(e)
        
        # Check if all original edges are covered
        # We need the original edges again
        edges = instance(n, k, p, seed)
        if edges != covered_edges:
            # Find missing
            missing = edges - covered_edges
            extra = covered_edges - edges
            if missing:
                # This would be an error in our logic
                pass
            if extra:
                raise ValueError("Clique contains non-edge")
        
        results.append(cliques)
    
    return results

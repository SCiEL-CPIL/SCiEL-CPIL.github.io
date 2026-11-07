import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        # K is 0..k-1, I is k..n-1
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def min_clique_cover_subgraph(nodes, edges):
        """
        Finds a minimum clique cover for a graph defined by 'nodes' and 'edges'.
        This is equivalent to finding the chromatic number of the complement graph.
        nodes: list of node IDs (integers)
        edges: set of tuples (u, v) with u < v
        Returns: list of cliques (lists of node IDs)
        """
        if not edges:
            return []
        
        # Build complement graph adjacency
        # Vertices involved in edges
        v_set = set()
        for u, v in edges:
            v_set.add(u)
            v_set.add(v)
        
        # If there are isolated nodes in the subgraph induced by edges, they don't need covering?
        # No, the input 'edges' are the ones to cover.
        # The graph H has vertex set v_set and edge set edges.
        # We want to cover all edges in 'edges' with cliques in H.
        # This is equivalent to coloring the complement graph \bar{H} induced by v_set?
        # Not exactly. The complement graph \bar{H} has edges between nodes NOT connected in H.
        # A clique in H is an independent set in \bar{H}.
        # So we want to partition the edges of H into cliques? No, partition the vertices?
        # "Clique cover" usually means covering the vertices with cliques.
        # But the task says "every edge must lie in exactly one listed clique".
        # This is an Edge Clique Cover.
        
        # Let's re-read carefully: "Every listed set must be a clique ... and every edge must lie in exactly one listed clique."
        # This is a partition of the edge set into cliques.
        
        # So for the remaining K-edges E_rem, we need to partition E_rem into cliques of K.
        # Since K is a clique, any subset of vertices forms a clique.
        # So we just need to partition the set of edges E_rem into subsets, such that each subset forms a clique in K.
        # Since K is a complete graph, ANY subset of edges forms a clique?
        # No. A set of edges forms a clique if the union of their endpoints is a clique in the graph.
        # In K, any set of vertices is a clique.
        # So any set of edges in K is a "clique" in the sense that the vertices involved form a clique.
        # Wait. If I have edges (0,1) and (1,2), the vertices are {0,1,2}. In K, {0,1,2} is a clique.
        # So the set of edges {(0,1), (1,2)} can be covered by the clique {0,1,2}.
        # In fact, ANY set of edges in a complete graph can be covered by the clique of all its vertices.
        # So the minimum number of cliques to cover E_rem is 1?
        # Just take the clique V_rem = {all vertices incident to E_rem}.
        # This single clique covers all edges in E_rem.
        
        # Is that allowed?
        # "Every edge must lie in exactly one listed clique."
        # If I list the clique V_rem, it contains all edges in E_rem.
        # Does it contain any edge NOT in E_rem?
        # The clique V_rem contains ALL pairs of vertices in V_rem.
        # Some of those pairs might be in E_covered (covered by cross-cliques).
        # If I list V_rem, those edges would be in TWO cliques: the cross-clique and V_rem.
        # This violates "exactly one".
        
        # So I cannot just take the union of vertices.
        # I need to partition the edges of E_rem into groups, such that for each group, the clique formed by its vertices does NOT contain any edge from E_covered.
        # Let G_rem = (V_rem, E_rem).
        # I need to partition E_rem into sets E_1, ..., E_m.
        # For each E_i, let V_i be the vertices incident to E_i.
        # The condition is: for all (u,v) in V_i x V_i, (u,v) must be in E_i.
        # This means E_i must be the set of ALL edges in G_rem induced by V_i.
        # So V_i must induce a subgraph in G_rem that is a clique? No.
        # The clique in the original graph K is V_i.
        # The edges covered by this clique are ALL pairs in V_i.
        # We require that ALL pairs in V_i are in E_i (and thus in E_rem).
        # This implies that V_i must be a clique in G_rem.
        # So we need to partition the edges of G_rem into cliques of G_rem.
        # This is the standard "Edge Clique Cover" problem?
        # No, it's partitioning the edge set into cliques.
        # This is equivalent to coloring the line graph L(G_rem)?
        # No, it's equivalent to finding a clique cover of the vertices of G_rem? No.
        
        # Let's look at it this way:
        # We want to write E_rem as a disjoint union of sets C_1, ..., C_m.
        # Each C_j is the edge set of a clique in K.
        # Since K is complete, C_j is the edge set of a complete subgraph on some vertex set V_j.
        # So C_j = { (u,v) | u,v in V_j, u<v }.
        # The condition "disjoint union" means no edge is in two C_j's.
        # The condition "covers E_rem" means every edge in E_rem is in some C_j.
        # The condition "C_j is a clique in K" is always true for any V_j.
        # BUT, we also need to ensure that we don't cover an edge that is NOT in E_rem.
        # If V_j contains an edge (u,v) that is in E_covered, then that edge would be covered by C_j.
        # But it was already covered by a cross-clique.
        # So we must ensure that for each V_j, ALL pairs in V_j are in E_rem.
        # This means V_j must be a clique in the graph G_rem = (V, E_rem).
        
        # So the problem reduces to:
        # Partition the edge set E_rem into a minimum number of cliques in G_rem.
        # This is the "Minimum Edge Clique Cover" problem?
        # No, standard Edge Clique Cover allows edges to be covered multiple times? No, usually it's a cover.
        # Here it is a PARTITION.
        # This is equivalent to coloring the vertices of the complement graph?
        # Let's think about the complement of G_rem within K.
        # Let H be the graph on V with edges E_covered.
        # Then E_rem = E(K) \ E(H).
        # We want to partition E_rem into cliques of G_rem.
        # A clique in G_rem is a set of vertices V_j such that all pairs in V_j are in E_rem.
        # This means NO pair in V_j is in H.
        # So V_j must be an independent set in H.
        # So we need to partition the edges of G_rem into sets that are complete graphs on independent sets of H.
        
        # This is a known problem?
        # Actually, since H is the union of cliques N(v), H is a specific type of graph.
        
        # Simple approach:
        # Greedy:
        # While E_rem is not empty:
        #   Find a maximal clique in G_rem.
        #   Remove its edges from E_rem.
        #   Add the clique to the cover.
        
        # Since k is small (<=30), we can find maximal cliques efficiently.
        # Or even better, since we just need a valid partition, any partition works.
        # To minimize, we want large cliques.
        # Maximal cliques in G_rem are good.
        
        # How to find maximal cliques in G_rem?
        # G_rem is a subgraph of K (clique).
        # So G_rem is just a set of edges.
        # We can use Bron-Kerbosch or just greedy.
        # Given the small size, a simple greedy "pick a vertex, add all neighbors in G_rem that are connected to all previously added" might work.
        
        # Let's implement a simple greedy for partitioning edges into cliques of G_rem.
        
        remaining_edges = set(edges)
        cliques = []
        
        # Order vertices by degree in G_rem?
        # Or just iterate.
        
        while remaining_edges:
            # Pick an arbitrary edge
            e = next(iter(remaining_edges))
            u, v = e
            
            # Start with clique V_c = {u, v}
            V_c = {u, v}
            
            # Try to add more vertices to V_c
            # A vertex w can be added if (w, x) is in remaining_edges for all x in V_c.
            # And w is not in V_c.
            
            # Candidates are vertices incident to edges in remaining_edges connected to V_c?
            # Actually, w must be connected to ALL vertices in V_c in G_rem.
            
            # Let's find all w such that w is connected to u and v in G_rem.
            # Then check if w is connected to all others in V_c.
            
            # To be efficient, we can just iterate through all vertices in the graph.
            # But we only care about vertices that can be part of a clique covering the current edge.
            
            # Let's collect candidates.
            candidates = set()
            for (a, b) in remaining_edges:
                if a in V_c or b in V_c:
                    if a not in V_c: candidates.add(a)
                    if b not in V_c: candidates.add(b)
            
            # Filter candidates
            valid_candidates = []
            for w in candidates:
                # Check if w is connected to all x in V_c in G_rem
                is_connected = True
                for x in V_c:
                    edge = (min(w, x), max(w, x))
                    if edge not in remaining_edges:
                        is_connected = False
                        break
                if is_connected:
                    valid_candidates.append(w)
            
            # To maximize the clique, we might want to add as many as possible.
            # This is the "Maximum Clique" problem on the induced subgraph?
            # No, we just want to extend V_c to a maximal clique in G_rem.
            # We can do this greedily.
            
            # Sort valid_candidates?
            # Just add them all if they form a clique among themselves?
            # No, we need to ensure the NEW set is a clique.
            # If we add w1 and w2, we need (w1, w2) in remaining_edges.
            
            # Simple greedy extension:
            # While we can add a vertex:
            #   Find a vertex w in valid_candidates that is connected to all current V_c.
            #   Add it to V_c.
            #   Update valid_candidates (remove w, and remove any w' not connected to w).
            
            # Actually, since we want to cover the edge e, we MUST include u and v.
            # The rest is optional but helps minimize count.
            
            # Let's try to find a maximal clique containing u and v in G_rem.
            # We can use a simple backtracking or just greedy.
            # Given small k, greedy is fine.
            
            # Greedy:
            # Current clique V_c = {u, v}
            # Available vertices: all w such that w is connected to all in V_c.
            # We iterate through all vertices in the graph (or just those incident to remaining edges).
            
            # To be safe and fast:
            # We'll just try to add vertices one by one.
            # This might not yield the MAXIMAL clique, but a maximal one is enough for a valid partition.
            # And larger cliques generally better.
            
            # Let's refine:
            # We want to find a clique in G_rem that contains u and v.
            # We can use a simple heuristic:
            # Order vertices by degree in G_rem (descending).
            # Try to add them if they connect to all current members.
            
            # Calculate degrees in G_rem for sorting?
            # Or just iterate through all possible vertices 0..k-1.
            
            potential_additions = []
            for w in range(k):
                if w in V_c:
                    continue
                # Check if w connects to all in V_c
                ok = True
                for x in V_c:
                    if (min(w, x), max(w, x)) not in remaining_edges:
                        ok = False
                        break
                if ok:
                    potential_additions.append(w)
            
            # Sort potential_additions by degree in G_rem?
            # Or just shuffle?
            # Let's sort by degree to prefer high-degree nodes.
            # Degree in G_rem:
            deg = {w: 0 for w in potential_additions}
            for w in potential_additions:
                for x in potential_additions:
                    if w != x and (min(w, x), max(w, x)) in remaining_edges:
                        deg[w] += 1
                # Also count connections to V_c? No, they already connect to V_c.
                # Just count connections among potential_additions.
                
            # Actually, simpler: just iterate through potential_additions in some order.
            # Let's use a simple greedy:
            # While potential_additions is not empty:
            #   Pick the first one?
            #   Add it to V_c.
            #   Remove it and all non-neighbors from potential_additions.
            
            # This is fast.
            while potential_additions:
                # Pick the one with highest degree in the subgraph induced by potential_additions?
                # Or just the first one.
                # Let's pick the one with highest degree in G_rem globally?
                # We already calculated deg.
                # Let's just pick the first one for simplicity, or random.
                # Random might help avoid local minima?
                # But we want to be deterministic for reproducibility?
                # The seed is fixed, so random is deterministic if we use a fixed seed for the solver.
                # But we want to minimize cliques.
                # Let's sort by degree descending.
                
                potential_additions.sort(key=lambda w: deg.get(w, 0), reverse=True)
                w = potential_additions[0]
                V_c.add(w)
                # Update potential_additions: remove w and any w' not connected to w
                new_potential = []
                for w2 in potential_additions[1:]:
                    if (min(w, w2), max(w, w2)) in remaining_edges:
                        new_potential.append(w2)
                potential_additions = new_potential
            
            # Now V_c is a clique in G_rem.
            # Remove all edges in V_c from remaining_edges.
            edges_to_remove = set()
            for i in range(len(V_c)):
                for j in range(i + 1, len(V_c)):
                    a, b = list(V_c)[i], list(V_c)[j]
                    edges_to_remove.add((min(a, b), max(a, b)))
            
            remaining_edges -= edges_to_remove
            cliques.append(list(V_c))
            
        return cliques

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        cliques = []
        
        # 1. Create cliques for each v in I
        # C_v = {v} U N(v)
        covered_edges = set()
        
        for v in range(k, n):
            N_v = neighbors[v]
            if not N_v:
                continue
                
            clique_v = [v] + N_v
            cliques.append(clique_v)
            
            # Mark edges in N(v) as covered
            # N(v) is a subset of K. All pairs in N(v) are covered.
            for i in range(len(N_v)):
                for j in range(i + 1, len(N_v)):
                    u, w = N_v[i], N_v[j]
                    covered_edges.add((min(u, w), max(u, w)))
        
        # 2. Find remaining edges in K
        # K is 0..k-1
        E_rem = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges:
                    E_rem.add((u, w))
        
        # 3. Partition E_rem into cliques
        if E_rem:
            rem_cliques = min_clique_cover_subgraph(list(range(k)), E_rem)
            for c in rem_cliques:
                # c is a list of vertices in K
                # Ensure it's a valid clique (it is, by construction)
                # And at least 2 vertices
                if len(c) >= 2:
                    cliques.append(c)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return empty? No, must be valid.
            # If we run out of time, we can't really do much.
            # But the algorithm is fast.
            pass
        
        cliques = solve_instance(*spec)
        # Convert to list of lists of ints
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        results.append(clean_cliques)
        
    return results

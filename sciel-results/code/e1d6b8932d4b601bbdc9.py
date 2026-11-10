import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency lists
        # Vertices 0..k-1 are the clique C
        # Vertices k..n-1 are the independent set I
        adj = [set() for _ in range(n)]
        
        cross_edges = []
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            if u < k and v >= k:
                cross_edges.append((v, u)) # (independent_vertex, clique_vertex)
            elif v < k and u >= k:
                cross_edges.append((u, v))
        
        # For each vertex i in I (k..n-1), its neighbors in C are adj[i]
        # A clique containing i must be of the form {i} U S where S is a subset of adj[i] that is a clique in C.
        # Since C is a clique, ANY subset of adj[i] forms a clique with i.
        # Thus, the maximal cliques containing i are just {i} U adj[i].
        # However, we might want smaller cliques if they cover edges more efficiently?
        # No, in set cover, larger sets are generally better unless they overlap heavily.
        # But wait, the greedy algorithm for set cover picks the set covering the most uncovered elements.
        # The universe is the set of cross-edges.
        # Each candidate clique Q (which is {i} U S) covers the cross-edges (i, c) for c in S.
        # Note: Edges within C are already covered by the "implicit" clique C? 
        # No, the problem states: "Every listed set must be a clique ... and every edge must lie in exactly one listed clique."
        # The seed program listed every edge as a clique of size 2.
        # We can list the large clique C = {0, ..., k-1} as one clique. This covers all edges within C.
        # Then we only need to cover the cross-edges.
        # The cross-edges are between I and C.
        # A clique containing a vertex i in I can contain at most one vertex from I (since I is independent).
        # So any clique covering a cross-edge (i, c) must be of the form {i} U S where S subset of C.
        # Since C is a clique, {i} U S is a clique iff S subset of N(i) intersect C.
        # So for each i in I, we can choose any subset S_i of N(i) to form a clique Q_i = {i} U S_i.
        # The union of all Q_i for i in I, plus the clique C, must cover all edges.
        # Edges within C are covered by C.
        # Cross-edges (i, c) are covered by Q_i if c is in S_i.
        # Edges within I: none exist.
        # So the problem reduces to: For each i in I, choose a subset S_i of N(i) such that for every cross-edge (i, c), c is in S_i.
        # This is trivially satisfied by setting S_i = N(i).
        # But we want to MINIMIZE the number of cliques.
        # We have 1 clique for C.
        # And we have cliques Q_i for each i in I?
        # Do we need a clique for every i in I?
        # If i has no cross-edges, we don't need a clique for it (since it has no edges to cover).
        # If i has cross-edges, we need at least one clique containing i to cover those edges.
        # Can one clique cover edges for multiple i's? No, because I is independent, so a clique can contain at most one vertex from I.
        # Therefore, for each i in I that has at least one cross-edge, we must have at least one clique containing i.
        # Let's call this clique Q_i.
        # Q_i must cover all cross-edges incident to i. So N(i) must be a subset of Q_i intersect C.
        # Since Q_i is a clique, Q_i intersect C must be a subset of N(i).
        # So Q_i intersect C must be exactly N(i).
        # Thus, the only possible clique containing i that covers all its edges is Q_i = {i} U N(i).
        # So the partition is uniquely determined?
        # Wait. "Every edge must lie in exactly one listed clique."
        # If we list C and all Q_i, does any edge lie in more than one?
        # Edges in C are in C. Are they in any Q_i?
        # Q_i = {i} U N(i). The edges in Q_i are (i, c) for c in N(i) and (c1, c2) for c1, c2 in N(i).
        # The edges (c1, c2) are edges in C.
        # So if we list C, the edges (c1, c2) are in C.
        # If we list Q_i, the edges (c1, c2) are also in Q_i.
        # This violates "exactly one".
        
        # So we cannot simply list C and all Q_i.
        # We need to partition the edges.
        # The edges within C must be covered by cliques that do NOT contain any vertex from I?
        # Or if they contain a vertex from I, they must not overlap with other cliques on those C-C edges.
        
        # Let's rethink.
        # We need a collection of cliques K_1, ..., K_m.
        # 1. Each K_j is a clique.
        # 2. For every edge e in E, there is exactly one j such that e is a subgraph of K_j (i.e., both endpoints of e are in K_j).
        
        # Consider an edge (c1, c2) with c1, c2 in C.
        # It must be covered by some clique K_j.
        # K_j contains c1 and c2.
        # If K_j contains any i in I, then i must be adjacent to both c1 and c2.
        # Also, K_j cannot contain any other i' in I (independent set).
        # So K_j is of the form {i} U S where S contains c1, c2.
        
        # Consider a cross-edge (i, c).
        # It must be covered by a clique K_l containing i and c.
        # K_l is of the form {i} U T where c is in T.
        
        # Key Insight:
        # For a fixed i in I, let N(i) be its neighbors in C.
        # Any clique containing i is of the form {i} U S where S subset N(i).
        # The edges covered by this clique are:
        # - Cross-edges (i, s) for s in S.
        # - C-edges (s1, s2) for s1, s2 in S.
        
        # If we use multiple cliques for the same i, say {i} U S1 and {i} U S2, 
        # then the cross-edges (i, s) for s in S1 are covered by the first, and for s in S2 by the second.
        # But S1 and S2 must be disjoint to avoid covering the same cross-edge twice?
        # If s is in S1 and S2, then edge (i, s) is in both cliques. Violation.
        # So for a fixed i, the sets S_j for all cliques containing i must be disjoint.
        # Also, the union of S_j must be N(i) to cover all cross-edges incident to i.
        
        # What about C-edges?
        # A C-edge (c1, c2) is covered by a clique {i} U S if c1, c2 in S.
        # It can also be covered by a clique {i'} U S' if c1, c2 in S'.
        # It can also be covered by a clique S_C that is a subset of C (no i).
        # To minimize the number of cliques, we should try to cover C-edges "for free" by the cliques we are already forced to create for the cross-edges.
        
        # Strategy:
        # 1. For each i in I with N(i) non-empty, we must cover all edges (i, c) for c in N(i).
        #    We can do this with one clique Q_i = {i} U N(i).
        #    This covers all cross-edges for i.
        #    It also covers all C-edges (c1, c2) where c1, c2 in N(i).
        # 2. If we do this for all i, we have |I_non_empty| cliques.
        # 3. Now, which C-edges are covered?
        #    A C-edge (c1, c2) is covered if there exists an i such that c1, c2 in N(i).
        #    Let U be the set of C-edges covered by at least one Q_i.
        #    Let R be the set of C-edges NOT covered by any Q_i.
        #    We must cover R with additional cliques.
        #    These cliques cannot contain any i in I, because:
        #       - If a clique K contains i, it is of the form {i} U S.
        #       - The C-edges it covers are within S.
        #       - But we already used Q_i = {i} U N(i).
        #       - If we add another clique {i} U S', S' must be disjoint from N(i) to not double-cover cross-edges.
        #       - But S' subset N(i) is required for it to be a clique containing i.
        #       - So S' must be empty. A clique of size 1 is not allowed (min 2 vertices).
        #       - So we cannot add more cliques containing i.
        #    Therefore, the cliques covering R must be subsets of C only.
        #    Since C is a clique, we can cover R with as few cliques as possible.
        #    Actually, since C is a clique, the entire set C is a clique.
        #    Can we just add C as a clique?
        #    If we add C, it covers ALL C-edges.
        #    But some C-edges are already covered by Q_i's.
        #    "Every edge must lie in exactly one listed clique."
        #    So if a C-edge is covered by Q_i, it cannot be in C.
        #    So we cannot add C if any C-edge is covered by a Q_i.
    
        # This implies we need to partition the C-edges.
        # Let E_C be the set of all edges in C.
        # For each i, let E_i be the set of C-edges (c1, c2) such that c1, c2 in N(i).
        # If we choose Q_i = {i} U N(i), we claim E_i.
        # We must ensure that E_i for different i's are disjoint?
        # No, if E_i and E_j overlap, an edge is covered by both. Violation.
        # So we can only choose Q_i if E_i is disjoint from E_j for all previously chosen Q_j?
        # Or we can modify S_i.
        # Instead of S_i = N(i), we can choose S_i subset N(i).
        # But we must cover all cross-edges (i, c). So N(i) must be covered by the union of S's for cliques containing i.
        # Since we can't have multiple cliques for i (as argued above, S' would have to be empty), we MUST have S_i = N(i) for the single clique Q_i containing i.
        # Wait, is it possible to have multiple cliques for i?
        # Clique 1: {i} U S1. Covers cross-edges (i, s) for s in S1.
        # Clique 2: {i} U S2. Covers cross-edges (i, s) for s in S2.
        # To cover all, S1 U S2 = N(i).
        # To avoid double covering cross-edges, S1 intersect S2 = empty.
        # So yes, we can split N(i) into multiple parts.
        # Why would we do that?
        # To reduce the C-edges covered by Q_i.
        # If S1 is small, E_{S1} is small.
        # This leaves more C-edges to be covered by other means.
        # But adding cliques increases the count.
        # Trade-off: 1 clique Q_i covering many C-edges vs 2 cliques covering fewer C-edges each.
        # Usually, 1 clique is better unless it causes a conflict that requires many extra cliques for the remaining C-edges.
        
        # Let's check the "conflict".
        # If E_i and E_j overlap, we have a problem.
        # Can we avoid the overlap?
        # We must include all c in N(i) in some clique for i.
        # If c1, c2 in N(i), the edge (c1, c2) is covered by the clique(s) for i that contain both.
        # If we split N(i) into S1, S2, then (c1, c2) is covered only if c1, c2 are in the same S.
        # So we can choose how many C-edges within N(i) are covered by i's cliques.
        # Ideally, we want to cover as many C-edges as possible to reduce the burden on "pure C" cliques.
        # But we must not double-cover.
        
        # Algorithm:
        # 1. Identify all i in I with N(i) non-empty.
        # 2. For each such i, we must create at least one clique containing i.
        #    Let's assume we create exactly one: Q_i = {i} U N(i).
        # 3. Check for conflicts: Do any two Q_i share a C-edge?
        #    If yes, we have a problem.
        #    However, in split graphs with p=0.7, N(i) is large.
        #    It is very likely that N(i) and N(j) share edges.
        #    Example: c1, c2 in N(i) AND c1, c2 in N(j).
        #    Then edge (c1, c2) is in Q_i and Q_j.
        #    This is invalid.
        
        # So we MUST resolve conflicts.
        # Option A: Split N(i) or N(j) so that the conflicting edge is not covered by both.
        # Option B: Don't use Q_i = {i} U N(i). Use smaller cliques.
        
        # Let's look at the structure of conflicts.
        # Edge (c1, c2) is covered by Q_i if c1, c2 in N(i).
        # Let H be the hypergraph where vertices are C-edges, and for each i, we have a hyperedge E_i (set of C-edges in N(i)).
        # We need to assign each C-edge to exactly one "owner".
        # Owners can be:
        # - A specific i (meaning the edge is covered by a clique containing i).
        # - "Pure C" (meaning the edge is covered by a clique subset of C).
        
        # If edge e is assigned to i, then e must be covered by a clique containing i.
        # This requires that the endpoints of e are in the same S_part of i.
        
        # This looks like a complex constraint satisfaction.
        
        # Simpler Heuristic:
        # 1. Start with Q_i = {i} U N(i) for all i.
        # 2. Find all C-edges covered by more than one Q_i.
        # 3. For each conflicting edge e=(c1, c2), it is covered by the set of i's such that c1, c2 in N(i).
        #    Let's pick one i to "keep" the edge, and for all other i's, we must ensure they do NOT cover e.
        #    To ensure Q_i' does not cover e, we must split N(i) such that c1 and c2 are in different parts.
        #    This increases the number of cliques for i.
        
        # Even simpler:
        # Just use the greedy set cover on the cross-edges, and handle C-edges separately?
        # No, the cliques for cross-edges inevitably cover some C-edges.
        
        # Let's try a different perspective.
        # Total cliques = (# cliques for I) + (# cliques for remaining C-edges).
        # If we use 1 clique per i, # cliques for I = |I_active|.
        # The C-edges covered are U = union of E_i.
        # The remaining C-edges are R = E_C \ U.
        # We need to cover R.
        # Since C is a clique, we can cover R with 1 clique if R is a clique?
        # No, R is a set of edges. The vertices involved in R form a graph G_R.
        # We need to partition the edges of G_R into cliques.
        # Since G_R is a subgraph of a clique, it's just a graph.
        # The minimum number of cliques to cover edges of a graph is the edge clique cover number.
        # For a general graph, this is NP-hard.
        # But G_R is a subgraph of a complete graph.
        # Actually, if we just add the set of vertices V_R (endpoints of edges in R) as a clique, it covers all edges in R.
        # Is V_R a clique? Yes, because it's a subset of C.
        # So we can cover all remaining C-edges with 1 clique: V_R.
        # BUT, we must ensure V_R does not contain any edge already covered by a Q_i.
        # Wait, if an edge is in R, it is NOT in U. So it is not covered by any Q_i.
        # So V_R covers exactly the edges in R.
        # Does V_R cover any edge in U?
        # If (c1, c2) is in U, it might be in V_R if c1, c2 are endpoints of some edge in R.
        # If (c1, c2) is in U and in V_R, then it's covered twice.
        # So we cannot just use V_R.
        
        # We need a clique K subset C such that E(K) is a subset of R.
        # And the union of E(K_j) for all such K_j is R.
        # To minimize the number of K_j, we want large cliques K_j whose edges are all in R.
        # This is equivalent to finding a clique cover of the graph G_R (where vertices are C and edges are R).
        # Since G_R is a subgraph of a clique, any subset of vertices is a clique.
        # So we just need to partition the edge set R into sets that form cliques?
        # No, a clique K covers all edges between its vertices.
        # So we need a collection of vertex sets K_1, ..., K_m such that:
        # 1. For each l, all edges between vertices in K_l are in R. (i.e., K_l is a clique in G_R)
        # 2. Every edge in R is covered by exactly one K_l.
        
        # This is the Edge Clique Partition problem.
        # For small n (30, 60, 90), we can use a greedy heuristic.
        # Greedy:
        # While R is not empty:
        #   Find the largest clique in G_R (largest number of edges).
        #   Add it to the partition.
        #   Remove its edges from R.
        
        # Now, back to the I part.
        # We assumed 1 clique per i.
        # If conflicts exist (edges in U covered by multiple i's), we have a problem.
        # How to handle?
        # If an edge e is covered by i and j, we must remove it from one of them.
        # Removing e from Q_i means we must split N(i) so that endpoints of e are separated.
        # This adds 1 clique for i (total 2 for i).
        # Is it worth it?
        # If we don't fix it, the result is invalid.
        # So we MUST fix it.
        
        # Revised Algorithm:
        # 1. Build G_R initially as all C-edges.
        # 2. For each i in I_active:
        #    Let N_i = N(i).
        #    We want to cover cross-edges of i.
        #    We want to cover as many C-edges within N_i as possible to remove them from G_R.
        #    But we cannot cover an edge if it's already "claimed" by another i?
        #    Actually, we can process i's one by one.
        #    For i:
        #       Find the largest subset S of N_i such that all edges in S are still in G_R.
        #       Wait, S must be a clique in G_R? No, S is a subset of vertices.
        #       The edges covered are E(S).
        #       We need E(S) to be a subset of current G_R.
        #       This means G_R must contain all edges between vertices in S.
        #       So S must be a clique in G_R.
        #       We want to maximize |E(S)| to reduce G_R.
        #       Also, we must cover all cross-edges of i.
        #       So we need to cover N_i with one or more such cliques S_1, ..., S_m.
        #       The union of S_j must be N_i.
        #       The S_j must be disjoint (to avoid double covering cross-edges? No, cross-edges are (i, c). If c is in S1 and S2, edge (i, c) is covered twice. So S_j must be disjoint).
        #       So we need to partition N_i into cliques S_1, ..., S_m of G_R.
        #       This is a Vertex Clique Partition of the induced subgraph G_R[N_i]?
        #       No, S_j must be a clique in G_R.
        #       So we need to partition the vertices N_i into cliques of G_R.
        #       This is the Clique Partition of the graph G_R[N_i].
        #       Greedy: Pick largest clique in G_R[N_i], remove its vertices, repeat.
        
        #    For each S_j found:
        #       Add clique {i} U S_j to the list.
        #       Remove all edges within S_j from G_R.
        
        # 3. After processing all i, G_R contains the remaining C-edges.
        # 4. Partition the edges of G_R into cliques.
        #    Greedy: Find largest clique in G_R, add it, remove its edges.
        
        # This seems robust and should produce a valid partition.
        # Complexity:
        # n=90, k=30.
        # C has 30 vertices. C-edges ~ 435.
        # I has 60 vertices.
        # For each i, N(i) has size ~ 21 (0.7 * 30).
        # Partitioning 21 vertices into cliques of G_R[N_i].
        # G_R is a subgraph of K_30.
        # Finding largest clique in a graph of 30 vertices is fast.
        # This should run well within 150s.

    def find_clique_partition(edge_set, vertices):
        """
        Partition the edge_set into cliques.
        vertices: set of vertices involved.
        Returns list of cliques (lists of vertices).
        """
        if not edge_set:
            return []
        
        # Build adjacency for the subgraph
        # We need to find cliques in the graph defined by edge_set.
        # Greedy: Repeatedly find the largest clique, remove its edges.
        
        remaining_edges = set(edge_set)
        cliques = []
        
        while remaining_edges:
            # Find largest clique in the graph defined by remaining_edges
            # Vertices are those incident to remaining_edges
            v_set = set()
            for u, v in remaining_edges:
                v_set.add(u)
                v_set.add(v)
            
            # Heuristic for Max Clique:
            # Since n is small (<=30 for C), we can use a simple greedy or even brute force if very small.
            # Greedy Max Clique:
            # Order vertices by degree in remaining graph.
            # Pick highest degree, add to clique, remove non-neighbors.
            
            # Let's use a simple recursive backtracking for max clique if v_set is small, or greedy.
            # Given the constraints, greedy is probably fine.
            
            # Build adjacency list for current remaining graph
            adj = {v: set() for v in v_set}
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Greedy Max Clique
            candidates = list(v_set)
            # Sort by degree descending
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            
            clique = []
            # This is a simple greedy, might not be optimal but good enough
            # Better: Bron-Kerbosch or similar?
            # For n=30, a simple recursive search for max clique is feasible.
            
            def max_clique_search(cands, current):
                if not cands:
                    return current
                # Pick a vertex
                v = cands[0]
                # Branch 1: include v
                new_cands = [u for u in cands[1:] if u in adj[v]]
                res1 = max_clique_search(new_cands, current + [v])
                # Branch 2: exclude v
                res2 = max_clique_search(cands[1:], current)
                return res1 if len(res1) > len(res2) else res2
            
            # To avoid exponential blowup, add a timeout or limit?
            # n=30, 2^30 is too big. But the graph is dense (subgraph of complete).
            # In dense graphs, max clique is large, so the tree is shallow.
            # Let's try it.
            
            best_clique = max_clique_search(candidates, [])
            
            if not best_clique:
                # Fallback: take any edge
                u, v = next(iter(remaining_edges))
                best_clique = [u, v]
            
            cliques.append(best_clique)
            
            # Remove edges covered by this clique
            for u in best_clique:
                for v in best_clique:
                    if u < v:
                        remaining_edges.discard((u, v))
        
        return cliques

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # C = 0..k-1, I = k..n-1
        C_vertices = set(range(k))
        I_vertices = set(range(k, n))
        
        # Identify C-edges
        C_edges = set()
        for u in range(k):
            for v in range(u+1, k):
                if v in adj[u]:
                    C_edges.add((u, v))
        
        # Initialize G_R as all C-edges
        G_R = set(C_edges)
        
        result_cliques = []
        
        # Process each i in I
        for i in sorted(I_vertices):
            if time.time() > DEADLINE:
                break # Safety break, though we should return best effort
            
            N_i = adj[i] & C_vertices
            if not N_i:
                continue
            
            # We need to partition N_i into cliques of G_R[N_i]
            # Greedy Vertex Partition into Cliques
            remaining_N = set(N_i)
            
            while remaining_N:
                # Find largest clique in G_R induced by remaining_N
                # Build subgraph
                sub_edges = set()
                for u in remaining_N:
                    for v in remaining_N:
                        if u < v and (u, v) in G_R:
                            sub_edges.add((u, v))
                
                if not sub_edges:
                    # No edges in G_R among remaining_N.
                    # We can't form a clique of size > 1 in G_R.
                    # But we need to cover cross-edges.
                    # If we pick a single vertex u, the clique is {i, u}. Size 2. Valid.
                    # It covers no C-edges.
                    # We must pick all remaining vertices as singletons?
                    # No, we can pick any subset. But to minimize cliques, we want large subsets.
                    # But if no edges in G_R, any subset of size > 1 is NOT a clique in G_R.
                    # So we MUST pick singletons.
                    # This means we add a clique {i, u} for each u in remaining_N.
                    # And we don't remove any edges from G_R.
                    for u in remaining_N:
                        result_cliques.append([i, u])
                    break
                
                # Find max clique in subgraph defined by sub_edges
                v_set = set()
                for u, v in sub_edges:
                    v_set.add(u)
                    v_set.add(v)
                
                adj_sub = {v: set() for v in v_set}
                for u, v in sub_edges:
                    adj_sub[u].add(v)
                    adj_sub[v].add(u)
                
                candidates = list(v_set)
                candidates.sort(key=lambda x: len(adj_sub[x]), reverse=True)
                
                def max_clique_search(cands, current):
                    if not cands:
                        return current
                    v = cands[0]
                    new_cands = [u for u in cands[1:] if u in adj_sub[v]]
                    res1 = max_clique_search(new_cands, current + [v])
                    res2 = max_clique_search(cands[1:], current)
                    return res1 if len(res1) > len(res2) else res2
                
                best_clique = max_clique_search(candidates, [])
                
                if not best_clique:
                    best_clique = [next(iter(remaining_N))]
                
                # Add clique {i} U best_clique
                result_cliques.append([i] + best_clique)
                
                # Remove edges within best_clique from G_R
                for u in best_clique:
                    for v in best_clique:
                        if u < v:
                            G_R.discard((u, v))
                
                # Remove best_clique from remaining_N
                remaining_N.difference_update(best_clique)
        
        # Process remaining C-edges in G_R
        if G_R:
            # Partition edges of G_R into cliques
            c_cliques = find_clique_partition(G_R, C_vertices)
            result_cliques.extend(c_cliques)
        
        # Convert to lists of ints
        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            # Fill remaining with trivial solution to avoid crash?
            # Or just return what we have?
            # The contract says "Return a list of three lists".
            # If we run out of time, we must return something valid.
            # Trivial solution: all edges as cliques of size 2.
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)
            
    return results

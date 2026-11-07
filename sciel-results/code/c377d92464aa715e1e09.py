"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    t_start = time.time()
    DEADLINE = 110.0  # safety margin before 150s limit

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        """Find a clique partition of the split graph with minimum cliques."""
        # Build the graph
        rng = Random(seed)
        # C = clique vertices 0..k-1, I = independent set vertices k..n-1
        # Edges: all C-C edges, cross edges with prob p
        # Build adjacency as sets
        adj = [set() for _ in range(n)]
        for u in range(k):
            for v in range(u + 1, k):
                adj[u].add(v)
                adj[v].add(u)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)

        # In a split graph, every maximal clique contains at least one vertex from C
        # (actually, since C is a clique, every maximal clique is C' ∪ I' where C' ⊆ C is a clique in C
        #  and I' are the I-vertices adjacent to all of C')
        # But since C is a complete graph, any subset of C is a clique.
        # So maximal cliques are: for each subset S of C, S ∪ {i in I : i adjacent to all of S}
        # But we want a partition, so we need to cover all edges with cliques that are vertex-disjoint.

        # Key insight: In a split graph, every edge is either C-C or C-I.
        # C-C edges: all edges within C.
        # C-I edges: edges between C and I.

        # Strategy: 
        # 1. For each vertex i in I, it can only appear in cliques with C-vertices it's adjacent to.
        # 2. We want to group C-I edges efficiently.
        # 3. C-C edges must be covered by cliques containing only C-vertices (or C+I but then the C part must be a clique).

        # A clique in a split graph is determined by its C-part (a subset of C) and its I-part (subset of I adjacent to all C-part).
        # Since C is complete, any subset of C works as the C-part.

        # To minimize the number of cliques:
        # - Each clique can cover many C-C edges (if it has many C-vertices) and many C-I edges.
        # - But cliques must be vertex-disjoint.

        # Greedy approach:
        # 1. First, handle C-I edges. For each i in I, find a clique that includes i and some C-vertices.
        #    But since cliques are vertex-disjoint, each C-vertex can only be in one clique.
        #    So we need to assign C-vertices to I-vertices (or to C-only cliques).

        # Better approach: Think of it as an edge coloring / partition problem.
        # Each edge must be in exactly one clique. Each clique is a set of vertices where every pair is an edge.

        # Let's use a different strategy:
        # 1. All C-C edges must be covered. Since C is a clique, we can put all C-vertices in one clique, 
        #    but then no I-vertex can be in that clique unless it's adjacent to ALL C-vertices.
        # 2. If we put all C in one clique, we cover all C-C edges with 1 clique, but then I-vertices 
        #    adjacent to all C can join, and others need separate cliques.

        # Let me think about this more carefully.
        # A clique partition: a set of cliques such that every edge is in exactly one clique and cliques are vertex-disjoint.

        # For split graphs, there's a known result: the minimum clique partition number.
        # Let's use a greedy algorithm:
        # 1. For each i in I (in some order), create a clique {i} ∪ N_C(i) where N_C(i) = {c in C : c adjacent to i}.
        #    But wait, this might use C-vertices that are needed for other purposes.
        # 2. After processing all I-vertices, remaining C-vertices form a clique (since C is complete).

        # But this doesn't minimize the number of cliques. We need to be smarter.

        # Alternative: 
        # - Group I-vertices that share similar C-neighborhoods.
        # - For a group of I-vertices I_g, if they all share a common C-neighborhood S_g, we can make one clique S_g ∪ I_g.
        # - But S_g must be a subset of C, and different cliques can't share C-vertices.

        # Let me try a different approach:
        # 1. Each C-vertex c can be assigned to at most one clique.
        # 2. Each I-vertex i must be in a clique that includes at least one C-vertex it's adjacent to (since I is independent, i can't be in a clique with another I-vertex unless... wait, I is independent so no two I-vertices are adjacent, so a clique can contain at most one I-vertex!

        # WAIT. This is crucial: In a split graph, the independent set I has no edges between its vertices.
        # So any clique can contain at most ONE vertex from I.
        # Therefore, each I-vertex must be in its own clique (or in a clique with C-vertices).
        # And each such clique has exactly one I-vertex plus some C-vertices.

        # So the problem reduces to:
        # - Assign each C-vertex to at most one I-vertex (or to a C-only clique).
        # - For each I-vertex i, its clique is {i} ∪ A_i where A_i ⊆ N_C(i) (C-vertices adjacent to i).
        # - The C-vertices not assigned to any I-vertex form one or more C-only cliques.
        # - We want to minimize total cliques = (number of I-vertices that have non-empty A_i) + (number of C-only cliques needed for unassigned C-vertices).
        #   Wait, actually every I-vertex needs a clique (since it has edges to C-vertices, those edges must be covered).
        #   But if an I-vertex i has no C-neighbors, it has no edges, so it doesn't need a clique.
        #   Actually, every edge must be in exactly one clique. An isolated vertex doesn't need to be in any clique.

        # Let me re-examine: We need to cover all edges. 
        # Edges are: C-C edges and C-I edges.
        # Each C-I edge (c, i) must be in a clique containing both c and i.
        # Since i can only be in one clique (because I is independent, i can't share a clique with another I-vertex),
        # all C-I edges incident to i must be in the same clique.
        # So for each i in I with at least one C-neighbor, we need a clique containing i and all its C-neighbors? 
        # No, we need a clique containing i and the C-vertices that form the C-I edges. But the clique must be a clique, 
        # so all C-vertices in the clique must be mutually adjacent (they are, since C is complete) and adjacent to i.
        # So the clique for i is {i} ∪ S_i where S_i ⊆ N_C(i).
        # But we need ALL C-I edges incident to i to be covered. So S_i must include ALL C-vertices adjacent to i.
        # Therefore, S_i = N_C(i), and the clique for i is {i} ∪ N_C(i).

        # Wait, but then C-vertices are used up. If c is in N_C(i) for multiple i's, it can only be in one clique.
        # So we have a conflict: c can only be assigned to one I-vertex's clique.

        # But the C-I edge (c, i) must be covered. If c is assigned to i' ≠ i, then edge (c, i) is not in i's clique.
        # So edge (c, i) must be in some other clique. But the only clique that can contain both c and i is a clique 
        # containing both, and since i can only be in one clique, that clique must be i's clique.
        # Contradiction! So actually, each C-I edge (c, i) forces c to be in i's clique.
        # Therefore, N_C(i) must be the exact set of C-vertices in i's clique, and no C-vertex can be adjacent to 
        # two different I-vertices... but that's not true in general.

        # I think I'm overcomplicating this. Let me re-read the problem.
        # "every edge must lie in exactly one listed clique"
        # So each edge is in exactly one clique. Cliques are vertex-disjoint (implied by partition).
        
        # For a C-I edge (c, i): it must be in a clique containing c and i.
        # Since i is in I (independent), i can only be in a clique with C-vertices.
        # So i's clique contains i and some C-vertices. All C-I edges incident to i must be in i's clique.
        # So i's clique = {i} ∪ N_C(i) where N_C(i) = all C-vertices adjacent to i.
        # But then if c is adjacent to both i and j (i ≠ j), c would need to be in both i's and j's cliques, 
        # which is impossible since cliques are vertex-disjoint.

        # This means: in a valid clique partition, no C-vertex can be adjacent to two different I-vertices?
        # That can't be right for general split graphs. Let me reconsider.

        # OH WAIT. I think I misunderstood. Let me re-read: "a clique on the first k vertices, an independent set on the rest"
        # So C = {0, ..., k-1} is a clique, I = {k, ..., n-1} is an independent set.
        # Cross edges are random with prob p.

        # In a clique partition, cliques are vertex-disjoint and cover all edges.
        # A clique can contain multiple C-vertices and at most one I-vertex (since I is independent).
        
        # For C-I edge (c, i): must be in a clique with both c and i.
        # So i's clique must include c.
        # If c is adjacent to i and j, then c must be in both i's and j's cliques. Impossible.
        
        # UNLESS... we don't require that all C-I edges incident to i are in i's single clique.
        # Wait, i can only be in ONE clique (since it's a partition). So all edges incident to i that are covered 
        # must be in i's clique. But i's edges are only to C-vertices. So all C-I edges incident to i must be in i's clique.
        # Therefore, i's clique = {i} ∪ {c in C : (c,i) is an edge}.
        # And this requires that no C-vertex is adjacent to two I-vertices.

        # But in our random split graph, C-vertices ARE adjacent to multiple I-vertices with high probability.
        # So how can a clique partition exist?

        # I think I'm misunderstanding the problem. Let me re-read the task.
        # "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
        
        # Maybe the cliques don't have to be vertex-disjoint? Let me re-read.
        # "Clique partitions" - a partition usually means disjoint. But maybe here it means a cover?
        # "every edge must lie in exactly one listed clique" - this means each edge is in exactly one clique, 
        # but vertices can be in multiple cliques?

        # If vertices can be in multiple cliques, then it's a clique cover (not partition).
        # Let me assume it's a clique cover where each edge is in exactly one clique, but vertices can repeat.

        # With that interpretation:
        # - C-C edges: can be covered by cliques of C-vertices.
        # - C-I edges: each (c, i) must be in a clique containing c and i.
        # - Since I is independent, a clique with an I-vertex i can only contain i and C-vertices adjacent to i.
        # - We can have a clique {c, i} for each C-I edge, but that's too many.
        # - Better: for each i, make a clique {i} ∪ N_C(i) which covers all C-I edges incident to i.
        #   But then C-C edges within N_C(i) are also covered by this clique.
        # - We need to cover C-C edges not covered by I-cliques.

        # Let me go with this interpretation and implement a greedy algorithm.

        # Build N_C(i) for each i in I
        N_C = [set() for _ in range(n)]
        for i in range(k, n):
            for c in range(k):
                if c in adj[i]:
                    N_C[i].add(c)

        # For each i in I with non-empty N_C(i), create clique {i} ∪ N_C(i)
        # Track which C-C edges are covered
        covered_cc = set()
        cliques = []
        for i in range(k, n):
            if N_C[i]:
                clique = [i] + sorted(N_C[i])
                cliques.append(clique)
                # Mark C-C edges within N_C(i) as covered
                nc_list = sorted(N_C[i])
                for a_idx in range(len(nc_list)):
                    for b_idx in range(a_idx + 1, len(nc_list)):
                        covered_cc.add((nc_list[a_idx], nc_list[b_idx]))

        # Now cover remaining C-C edges
        # Remaining C-C edges are those not in covered_cc
        # We can group them into cliques
        # Since C is a clique, any subset of C is a clique
        # Greedy: find maximal cliques in the remaining C-C edge graph
        # The remaining C-C graph has vertices 0..k-1 and edges = all C-C edges minus covered_cc
        
        remaining_cc_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_cc:
                    remaining_cc_edges.append((u, v))

        if remaining_cc_edges:
            # Build adjacency for remaining C-C graph
            rem_adj = [set() for _ in range(k)]
            for u, v in remaining_cc_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Greedy: repeatedly find a maximal clique
            unvisited = set(range(k))
            while unvisited:
                # Pick a vertex with highest degree in remaining graph
                best_v = max(unvisited, key=lambda x: len(rem_adj[x] & unvisited))
                if not (rem_adj[best_v] & unvisited):
                    # Isolated vertex, no edges to cover
                    unvisited.discard(best_v)
                    continue
                # Greedy maximal clique: start with best_v, add neighbors
                clique_set = {best_v}
                candidates = rem_adj[best_v] & unvisited
                while candidates:
                    # Pick vertex with most neighbors in candidates
                    best_c = max(candidates, key=lambda x: len(rem_adj[x] & candidates))
                    clique_set.add(best_c)
                    candidates = candidates & rem_adj[best_c]
                if len(clique_set) >= 2:
                    cliques.append(sorted(clique_set))
                unvisited -= clique_set
                # Remove edges covered by this clique
                cl_list = sorted(clique_set)
                for a_idx in range(len(cl_list)):
                    for b_idx in range(a_idx + 1, len(cl_list)):
                        u, v = cl_list[a_idx], cl_list[b_idx]
                        if u in rem_adj and v in rem_adj[u]:
                            rem_adj[u].discard(v)
                            rem_adj[v].discard(u)

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            # Return what we have with a valid fallback
            break
        cliques = solve_one(*spec)
        results.append(cliques)
    
    # Ensure we have 3 results
    while len(results) < 3:
        spec = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[len(results)]
        cliques = solve_one(*spec)
        results.append(cliques)

    # Convert to ensure proper types
    final = []
    for cliques in results:
        converted = []
        for clique in cliques:
            if isinstance(clique, (list, tuple)):
                converted.append([int(v) for v in clique])
            else:
                converted.append([int(v) for v in clique])
        final.append(converted)
    
    return final
# EVOLVE-BLOCK-END

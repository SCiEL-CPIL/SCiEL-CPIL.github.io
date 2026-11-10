"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def bipartite_edge_color(adj, left_nodes, right_nodes):
        """
        Edge-color a bipartite graph. adj[c] = list of independent neighbors of clique vertex c.
        Returns list of color classes; each color class is a list of (c, s) edges forming a matching.
        """
        # Build adjacency
        # left_nodes = clique vertices [0..k-1], right_nodes = independent [k..n-1]
        # We'll do standard edge coloring: repeatedly find a matching in the uncolored edges
        # and assign a new color.
        
        # For efficiency, use the fact that bipartite edge chromatic number = max degree
        # We'll find matchings iteratively
        
        remaining = defaultdict(list)
        for c in left_nodes:
            for s in adj.get(c, []):
                remaining[c].append(s)
        
        # Also track for each right node which left nodes still connect
        remaining_right = defaultdict(list)
        for c in left_nodes:
            for s in remaining[c]:
                remaining_right[s].append(c)
        
        color_classes = []
        
        while True:
            # Check if any edges remain
            total_remaining = sum(len(v) for v in remaining.values())
            if total_remaining == 0:
                break
            
            # Find a maximal matching in the remaining graph
            # Greedy matching: for each left node, try to match to an available right node
            matched_left = set()
            matched_right = set()
            matching = []
            
            # Process left nodes in order of remaining degree (highest first) for better matchings
            left_order = sorted(left_nodes, key=lambda c: -len(remaining[c]))
            
            for c in left_order:
                if c in matched_left:
                    continue
                # Try to find an available right neighbor
                for s in remaining[c]:
                    if s not in matched_right:
                        matching.append((c, s))
                        matched_left.add(c)
                        matched_right.add(s)
                        break
            
            if not matching:
                break
            
            color_classes.append(matching)
            
            # Remove matched edges
            for c, s in matching:
                remaining[c].remove(s)
                remaining_right[s].remove(c)
        
        return color_classes

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate clique-part edges and cross-edges
        clique_edges = []
        cross_edges = []
        for u, v in edges:
            if u < k and v < k:
                clique_edges.append((u, v))
            else:
                cross_edges.append((u, v))
        
        # Build adjacency for bipartite graph
        adj = defaultdict(list)
        for u, v in cross_edges:
            adj[u].append(v)
        
        left_nodes = list(range(k))
        
        # Edge-color the bipartite graph
        color_classes = bipartite_edge_color(adj, left_nodes, list(range(k, n)))
        
        # Now assign each color class (matching) to a clique vertex
        # We need to assign matching M to vertex c such that deg(c) >= |M|
        # Actually, the clique is {c} union {s : (c,s) in M}, and this is valid as long as
        # all s in M are neighbors of c (which they are, since M is a matching from the edge set)
        # And no two matchings assigned to different c's share an s vertex (guaranteed since each
        # color class is a matching, and different color classes are edge-disjoint, but could they
        # share a vertex? No! In a proper edge coloring, each vertex has at most one edge of each color,
        # so each vertex appears in at most one edge per color class. But across different color classes,
        # a vertex can appear in multiple. Wait, that's fine - the constraint is that for a given independent
        # vertex s, it can only be in ONE clique total (since if s is in clique {c1}∪S1 and clique {c2}∪S2,
        # then edge (c1,s) is in both cliques, violating "exactly one").
        
        # So the constraint is: each independent vertex s appears in at most one color class total.
        # But in a proper edge coloring, s has degree d(s), so it appears in exactly d(s) color classes!
        # This is a problem.
        
        # I need to reconsider. The issue is that in a standard edge coloring, each vertex of degree d
        # appears in d different color classes. But we need each independent vertex to appear in at most
        # ONE clique.
        
        # So this approach doesn't work directly. Let me think again.
        
        # Alternative: each clique is {c} ∪ S where S ⊆ N(c). The constraint is that for each s,
        # s appears in at most one S. And for each edge (c,s), it must be in exactly one clique, which
        # means s must be in S_c (the set assigned to c).
        
        # So we need to partition the independent vertices into sets S_0, S_1, ..., S_{k-1} (one per c)
        # such that S_c ⊆ N(c) for all c, and every edge (c,s) is covered, i.e., s ∈ S_c for all (c,s) ∈ E_cross.
        
        # Wait, that means s must be in S_c for EVERY c that is adjacent to s. But s can only be in ONE S_c!
        # So this is only possible if each independent vertex s has degree at most 1!
        
        # That's not true in general. So my initial formulation was wrong.
        
        # Let me re-examine. A clique in the split graph can include multiple clique vertices and multiple
        # independent vertices. If a clique contains c1, c2 (both in C) and s1, s2 (both in I), then it covers
        # edges (c1,s1), (c1,s2), (c2,s1), (c2,s2), and (c1,c2). The (c1,c2) edge is also in the big K_k clique,
        # so that would be double-covered. So a clique can contain at most ONE clique vertex if we already have
        # the big K_k clique.
        
        # So cliques are of the form {c} ∪ S where S ⊆ I, and S ⊆ N(c). The constraint is:
        # 1. For each c, the union of all S's assigned to c must cover N(c) (all neighbors of c).
        # 2. Each s ∈ I appears in at most one S total (across all c's).
        
        # So we need to assign each independent vertex s to exactly one clique vertex c that is adjacent to s,
        # and then for each c, S_c = {s : s assigned to c}. This covers edge (c,s) iff s is assigned to c.
        # But we need ALL edges (c,s) to be covered, so for each edge (c,s), s must be assigned to c.
        # But s can only be assigned to ONE c. So if s is adjacent to c1 and c2, we can't cover both (c1,s) and (c2,s).
        
        # This is impossible unless each s has degree ≤ 1!
        
        # I think I'm misunderstanding the problem. Let me re-read.
        
        # "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
        
        # So we need a collection of cliques such that every edge is in exactly one clique. The cliques can overlap
        # in vertices, but no edge can be in two cliques.
        
        # A clique {c1, c2, s1} contains edges (c1,c2), (c1,s1), (c2,s1). If we also have clique {c1, c2, s2},
        # it contains (c1,c2) again! So (c1,c2) is in two cliques. Violation.
        
        # So we can't have two cliques that both contain the same pair of clique vertices.
        
        # This means: for each pair {c1,c2} ⊆ C, at most one clique can contain both c1 and c2.
        
        # The big K_k clique contains all pairs. So no other clique can contain two clique vertices.
        
        # Therefore, all other cliques have exactly one clique vertex and some independent vertices.
        
        # So my formulation was correct, and it's impossible to cover all cross-edges if any independent
        # vertex has degree > 1.
        
        # Wait, that can't be right. The problem says it's always possible (with n^2/6 + O(n) cliques).
        
        # Let me reconsider. Maybe the cliques don't all have to be of the form {c} ∪ S. Maybe some cliques
        # have only independent vertices? No, independent vertices have no edges between them, so a clique
        # of only independent vertices has at most 1 vertex, but we need at least 2.
        
        # Maybe some cliques have only clique vertices? Yes, but those are subsets of K_k.
        
        # I think the issue is that I'm forcing the big K_k clique. Maybe it's better to NOT use the big
        # K_k clique, and instead use smaller cliques that each contain some clique vertices and some
        # independent vertices, such that no edge is repeated.
        
        # Let me try a different approach. For the cross-edges, we can group them into cliques where each
        # clique is {c} ∪ S_c for a single c, with S_c ⊆ N(c), and the S_c's are disjoint (each s in at most one S_c).
        # This covers edges (c,s) for s ∈ S_c. To cover ALL cross-edges, we need for each c, the union of S_c's
        # (over all cliques assigned to c) to equal N(c). But since S_c's are disjoint across all cliques,
        # each s is in at most one S_c total. So for s adjacent to c1 and c2, s can only be in S_{c1} or S_{c2},
        # not both. So we can only cover one of (c1,s) or (c2,s).
        
        # This is still impossible.
        
        # I must be misunderstanding the problem. Let me re-read very carefully.
        
        # "a clique on the first k vertices, an independent set on the rest and each cross pair joined with probability p"
        
        # So the graph has: C = {0,...,k-1} is a clique, I = {k,...,n-1} is an independent set, and each (c,s) 
        # with c∈C, s∈I is an edge with probability p.
        
        # "Every listed set must be a clique with at least two vertices and every edge must lie in exactly one listed clique."
        
        # So we need a set of cliques (each is a set of vertices that form a clique in the graph) such that
        # every edge of the graph is contained in exactly one of these cliques.
        
        # A clique in this graph can be:
        # - A subset of C (any size ≥ 2)
        # - {c} ∪ S where c ∈ C, S ⊆ I, and S ⊆ N(c) (since s1,s2 ∈ I have no edge, so at most... wait,
        #   {c, s1, s2} is a clique iff (c,s1) and (c,s2) are edges AND (s1,s2) is an edge. But s1,s2 ∈ I,
        #   so (s1,s2) is NOT an edge. So {c, s1, s2} is NOT a clique!
        
        # So a clique can contain at most ONE independent vertex! Because any two independent vertices are
        # not adjacent.
        
        # So cliques are either:
        # - Subsets of C (size ≥ 2)
        # - {c, s} where c ∈ C, s ∈ I, and (c,s) is an edge (size 2)
        # - {c} alone (size 1, but we need size ≥ 2, so not allowed)
        
        # So the only cliques of size ≥ 2 are subsets of C and edges {c,s} where (c,s) is a cross-edge.
        
        # This means: every cross-edge (c,s) must be in its own clique {c,s} (since no larger clique can contain
        # it, as adding another vertex would break the clique property).
        
        # And the clique-part edges can be covered by subsets of C.
        
        # So the minimum number of cliques is: (minimum number of subsets of C to cover all edges of K_k) + (number of cross-edges).
        
        # The minimum number of subsets of C to cover all edges of K_k with each edge in exactly one subset is...
        # This is equivalent to edge-coloring K_k! Each color class is a matching, and we need to cover all edges.
        # The edge chromatic number of K_k is k-1 if k is even, k if k is odd.
        
        # Wait, but the subsets don't have to be matchings. A subset {c1,c2,c3} covers edges (c1,c2), (c1,c3), (c2,c3).
        # So we can use larger cliques.
        
        # The problem is: partition the edges of K_k into cliques (subsets of C), minimizing the number of cliques.
        # Each clique of size m covers C(m,2) edges.
        
        # To minimize the number of cliques, we want to use as large cliques as possible. The largest clique is C itself,
        # which covers all edges. So we can use just ONE clique: C itself.
        
        # So the minimum is 1 + (number of cross-edges).
        
        # That matches the observed improvement: 1 -> 1.336, where the seed was using k(k-1)/2 + E_cross.
        
        # So the answer is simply: one big clique C, and one clique {c,s} for each cross-edge (c,s).
        
        # Let me verify: 
        # - Clique C = {0,...,k-1}: covers all edges within C. ✓
        # - Clique {c,s} for each cross-edge: covers each cross-edge exactly once. ✓
        # - No edge is in two cliques: C only has C-C edges, {c,s} only has the one cross-edge. ✓
        # - All cliques have size ≥ 2: C has size k ≥ 2 (k=10,20,30), {c,s} has size 2. ✓
        
        # So the total number of cliques is 1 + E_cross.
        
        # This is already optimal, since each cross-edge must be in its own clique (as argued above).
        
        # Let me implement this.
        
        # Count cross-edges
        num_cross = len(cross_edges)
        
        # Build the list of cliques
        cliques = [list(range(k))]  # the big clique
        for u, v in cross_edges:
            cliques.append([u, v])
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Return best result found so far (shouldn't happen for this simple algorithm)
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END

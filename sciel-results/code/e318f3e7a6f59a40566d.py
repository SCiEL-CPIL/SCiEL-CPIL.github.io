"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

N = 16
DEADLINE = 140.0  # safety margin before 150s kill

def _all_pairs():
    return [(i, j) for i in range(N) for j in range(i+1, N)]

def _precompute_q3s():
    """
    Precompute all subsets of 8 vertices that form a Q3.
    A Q3 is the 3-dimensional hypercube: 8 vertices, 12 edges.
    Vertices can be labeled 0..7 (binary 000..111).
    Edges connect vertices differing in exactly one bit.
    We need to find all injective mappings from {0..7} to {0..15} that form a Q3.
    To avoid duplicates, we fix the image of vertex 0 to be the smallest vertex in the subset.
    """
    # Q3 adjacency: vertex v is connected to v^1, v^2, v^4
    q3_adj = {}
    for v in range(8):
        q3_adj[v] = [v^1, v^2, v^4]
    
    q3_sets = []  # list of frozensets of 8 vertex indices
    
    # We iterate over all 8-subsets of 16 vertices
    # For each, check if it induces a Q3
    # But that's C(16,8) = 12870 subsets, and checking each is expensive.
    # Better: enumerate all permutations? No, 16P8 is too large.
    
    # Alternative: for each 8-subset, check if the induced subgraph is isomorphic to Q3.
    # Q3 has 12 edges. So the induced subgraph must have exactly 12 edges and be 3-regular.
    
    # Let's precompute all 8-subsets and check.
    for subset in itertools.combinations(range(N), 8):
        s = set(subset)
        # Count edges within the subset
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                # We need to know if (subset[i], subset[j]) is an edge in the "complete" sense?
                # No, we're checking if the INDUCED subgraph on this vertex set is a Q3.
                # But we don't have a fixed graph yet. We're looking for vertex sets that COULD be a Q3.
                # Actually, any 8 vertices can potentially form a Q3 if the right edges are present.
                # The task is: the graph must NOT contain a subgraph isomorphic to Q3.
                # So we need to check, for a given edge set, whether there exists a set of 8 vertices
                # such that the edges among them form exactly the Q3 edge pattern.
                
                # This approach of precomputing "Q3 vertex sets" doesn't work directly because
                # the Q3 is defined by the edges, not just the vertices.
                
                # Let me rethink.
                # A subgraph isomorphic to Q3 means: there exist 8 distinct vertices v0..v7 and
                # the edges {vi, vj} for all pairs differing in one bit are ALL present in the graph.
                # (Other edges among these 8 vertices may or may not be present; the subgraph just
                #  needs to CONTAIN the Q3 edges.)
                
                # So we need to check: does there exist an injective mapping f: {0..7} -> {0..15}
                # such that for all v in 0..7 and all bits b in {1,2,4}, the edge (f(v), f(v^b)) is in E?
                
                # This is equivalent to: does the graph contain a subgraph that has Q3 as a subgraph?
                # i.e., is there a set of 12 edges forming a Q3?
                
                # Number of possible Q3 embeddings: 16P8 / (symmetries of Q3) ... still large.
                # 16P8 = 16!/8! = 518,918,400. Too many to check naively.
                
                pass
        # This approach is too slow. Let's use a different strategy.
    
    return q3_sets  # empty for now

def _has_q3(edges_set, edge_lookup):
    """
    Check if the graph with edges in edges_set contains a Q3.
    edge_lookup: set of frozenset({u,v}) for O(1) edge check.
    """
    # We need to find 8 vertices that form a Q3.
    # Strategy: iterate over all 4-cycles? No, Q3 has many 4-cycles.
    
    # Better: use the fact that Q3 is bipartite with parts of size 4.
    # The two parts are the even-weight and odd-weight binary vectors.
    # So we need two sets A and B of 4 vertices each, such that:
    # - Each vertex in A is connected to exactly 3 vertices in B (in the Q3 subgraph)
    # - Each vertex in B is connected to exactly 3 vertices in A
    # - The induced bipartite subgraph between A and B is the 4x4 grid of Q3.
    
    # Actually, let's just enumerate all 8-subsets and check if they can form a Q3.
    # C(16,8) = 12870. For each, we need to check if there's a bijection to Q3 vertices
    # such that all 12 Q3 edges are present.
    
    # For a given 8-set S, we can try to match it to Q3.
    # Q3 has 8 vertices, 12 edges, each vertex degree 3.
    # So in the induced subgraph on S, each vertex must have degree at least 3 within S
    # (since the Q3 edges must be a subset of the induced edges).
    
    # Quick filter: if any vertex in S has < 3 neighbors in S, skip.
    
    # For the remaining candidates, we need to check if the induced subgraph CONTAINS a Q3.
    # This is a subgraph isomorphism problem, which is NP-hard in general, but Q3 is small.
    
    # Let's enumerate all permutations of S assigned to Q3 vertices 0..7.
    # But 8! = 40320 per 8-set, times 12870 sets = 519M. Too slow.
    
    # Better: use backtracking with pruning.
    # Or: precompute all possible Q3 edge patterns and check.
    
    # Alternative approach: enumerate all 12-edge subsets that could be a Q3.
    # A Q3 has 12 edges. Number of ways to choose 12 edges from C(16,2)=120 is C(120,12) ~ 10^18. Too many.
    
    # Let me think differently.
    # We can iterate over all ways to choose 8 vertices and check if the induced subgraph
    # contains a Q3. To speed up, we can use the following:
    # - In Q3, there are 6 faces (4-cycles). 
    # - Each pair of opposite faces are disjoint.
    
    # Actually, the most practical approach for 16 vertices:
    # Enumerate all 8-subsets (12870 of them).
    # For each, check if the induced subgraph has a Q3 as a subgraph.
    # To check this, we can use the fact that Q3 is the only 3-regular bipartite graph
    # on 8 vertices with girth 4... no, there are others.
    
    # Let's just do a smart backtracking for each 8-set.
    
    # Actually, let me try a different angle.
    # The number of Q3 subgraphs in K16 is finite. Let's precompute all of them.
    # A Q3 is determined by choosing 8 vertices and a bijection to {0..7} such that
    # the edges match. The number of distinct Q3 subgraphs in K16 is:
    # C(16,8) * 8! / |Aut(Q3)| = 12870 * 40320 / 48 = 10,767,360.
    # That's ~10M. We can precompute all of them!
    
    # Wait, that's the number of labeled Q3s. But many will be the same set of edges.
    # Actually, each Q3 subgraph is a specific set of 12 edges.
    # Different bijections may give the same edge set.
    
    # Let me just enumerate all 8-subsets and for each, find all Q3 embeddings.
    
    pass

def _build_edge_lookup(edges):
    return set(frozenset((u, v)) for u, v in edges)

def _count_q3_violations(edges_set, edge_lookup):
    """Count the number of Q3 subgraphs in the graph. Return 0 if Q3-free."""
    # For now, return a placeholder. We'll implement properly.
    return 0

def construct():
    start_time = time.time()
    
    # Precompute all Q3 subgraphs in K16
    # A Q3 subgraph is a set of 12 edges on 8 vertices.
    # We enumerate all 8-subsets of vertices, and for each, check all bijections to Q3.
    
    # Q3 vertices: 0..7, edges: (v, v^1), (v, v^2), (v, v^4) for v in 0..7, but each edge counted once.
    q3_edges = set()
    for v in range(8):
        for b in [1, 2, 4]:
            e = frozenset((v, v ^ b))
            q3_edges.add(e)
    # q3_edges has 12 elements (each undirected edge once)
    
    # Now, for each 8-subset of {0..15}, and for each permutation of it assigned to 0..7,
    # check if the 12 edges are all present. But we're building the graph, so we want to
    # PRECOMPUTE all possible Q3 edge sets in K16.
    
    # A Q3 edge set in K16 is a set of 12 edges {frozenset({f(v), f(v^b)}) : v in 0..7, b in {1,2,4}}
    # for some injective f: {0..7} -> {0..15}.
    
    # To avoid duplicates, we can canonicalize: the edge set is the same regardless of labeling.
    # So we just need to enumerate all distinct 12-edge sets that form a Q3.
    
    # Number of distinct Q3 edge sets in K16:
    # Each is determined by the image of f, which is an 8-subset, and the bijection.
    # But different bijections on the same 8-set can give different edge sets.
    
    # Let's just enumerate all 8-subsets and all 8! bijections, collect the edge sets, and deduplicate.
    # 12870 * 40320 = 519M. Too many.
    
    # Better: for each 8-subset S, the number of distinct Q3 edge sets induced by bijections is
    # at most 8! / |Aut(Q3)| = 40320/48 = 840. So total is at most 12870 * 840 = 10.8M.
    # Still a lot, but we can do it once at startup if we're fast.
    
    # Actually, let's just check on-the-fly during the search. The search will be local,
    # so we only need to check if ADDING an edge creates a new Q3.
    
    # When we add edge (u,v), we need to check if there exists a Q3 that contains (u,v).
    # A Q3 containing edge (u,v): u and v must be adjacent in Q3, so they differ in one bit.
    # WLOG, map u->0, v->1 (differ in bit 0). Then the other 6 vertices must be mapped to
    # 2,3,4,5,6,7 in some way, and all Q3 edges must be present.
    
    # For a fixed edge (u,v), the number of possible Q3s containing it is:
    # Choose which bit they differ in (3 choices), then assign the remaining 6 Q3 vertices
    # to 6 of the remaining 14 graph vertices: P(14,6) = 14!/8! = 2,162,160.
    # For each such assignment, check if all 12 edges are present.
    # So ~3 * 2.16M = 6.5M checks per edge addition. Too slow.
    
    # Let me think of a better approach.
    
    # Alternative: use the bipartite structure of Q3.
    # Q3 is bipartite with parts A (even parity) and B (odd parity), each of size 4.
    # Edges only between A and B.
    # So a Q3 subgraph is determined by:
    # - A 4-set A' of vertices (the "even" part)
    # - A 4-set B' of vertices (the "odd" part)
    # - A perfect matching structure: each a in A' connects to exactly 3 vertices in B',
    #   and the bipartite graph is the 4x4 Q3 bipartite graph.
    
    # The Q3 bipartite graph between A and B:
    # A = {0,3,5,6} (even parity), B = {1,2,4,7} (odd parity)
    # Edges:
    # 0-1, 0-2, 0-4
    # 3-2, 3-1, 3-7
    # 5-4, 5-1, 5-7
    # 6-7, 6-2, 6-4
    
    # So for each a in A, the 3 neighbors in B are determined.
    
    # To check if (A', B') forms a Q3:
    # 1. |A'| = 4, |B'| = 4
    # 2. There exists a bijection f: A' -> A and g: B' -> B such that
    #    for all a in A', b in B': edge (f(a), g(b)) in Q3 iff edge (a', b') in graph.
    
    # This is still complex. Let me just implement a direct check for small cases.
    
    # Given the time constraints, let me use a simpler but effective approach:
    # Start with a dense graph, then remove edges to eliminate Q3s.
    
    # Initial graph: K16 (120 edges)
    # We'll use a local search: start with K16, repeatedly remove an edge that is in
    # the most Q3s, until no Q3s remain. Then try to add edges back.
    
    # But first, we need an efficient Q3 checker.
    
    # Let me precompute all Q3 edge sets in K16.
    # I'll do it once, storing them as frozensets of frozensets (edge sets).
    
    q3_edge_sets = set()
    
    # To enumerate: for each 8-subset S of vertices, for each way to assign S to Q3 vertices 0..7
    # (i.e., each permutation of S), check if the induced edge set is a Q3.
    # But we only need to store the edge set if it's a valid Q3.
    
    # Optimization: two permutations that differ by an automorphism of Q3 give the same edge set.
    # So we can fix one vertex: map the smallest vertex in S to Q3 vertex 0.
    # Then we have 7! = 5040 permutations per 8-set.
    # 12870 * 5040 = 64.9M. Still a lot, but let's try with a time limit.
    
    # Actually, let's just do it. It's a one-time cost.
    
    import itertools
    
    all_q3s = []
    
    # Q3 edges in terms of vertex labels 0..7
    q3_adj = {v: [v^1, v^2, v^4] for v in range(8)}
    
    for S in itertools.combinations(range(N), 8):
        if time.time() - start_time > DEADLINE - 20:
            break
        s_set = set(S)
        # Fix: map S[0] (smallest) to Q3 vertex 0
        # Permute the remaining 7 elements to Q3 vertices 1..7
        for perm in itertools.permutations(S[1:]):
            if time.time() - start_time > DEADLINE - 20:
                break
            mapping = [S[0]] + list(perm)
            # Check if this mapping gives a valid Q3: all 12 edges must be within the vertex set
            # (which they are, since mapping is a permutation of S)
            # The edge set is determined by the mapping.
            edge_set = frozenset(
                frozenset((mapping[v], mapping[w]))
                for v in range(8)
                for w in q3_adj[v]
                if v < w
            )
            if len(edge_set) == 12:
                all_q3s.append(edge_set)
    
    # Deduplicate
    all_q3s = list(set(all_q3s))
    
    print(f"Precomputed {len(all_q3s)} Q3 edge sets", file=sys.stderr)
    
    # Now, start with K16 and remove edges to eliminate all Q3s.
    # Greedy: repeatedly find a Q3, remove one of its edges, repeat.
    
    edges = set(frozenset((i, j)) for i in range(N) for j in range(i+1, N))
    edge_list = list(edges)
    
    # To speed up, maintain a count of how many Q3s each edge is in.
    # But that's expensive to maintain. Let's just do a simple approach.
    
    # Simple approach: while there exists a Q3, pick one and remove a random edge from it.
    # To find a Q3, iterate over all_q3s and check if all 12 edges are present.
    
    # This is O(len(all_q3s) * 12) per check, which is ~10M * 12 = 120M. Too slow per iteration.
    
    # Better: for each edge, precompute which Q3s it belongs to.
    edge_to_q3s = {e: [] for e in edge_list}
    for q3 in all_q3s:
        for e in q3:
            edge_to_q3s[e].append(q3)
    
    # Now, when we remove an edge, we can update the "active" Q3s.
    # A Q3 is "active" if all 12 of its edges are present.
    
    # Let's maintain a set of active Q3s.
    active_q3s = set(all_q3s)
    
    # When we remove edge e, all Q3s containing e become inactive.
    # When we add edge e, Q3s containing e may become active if all other 11 edges are present.
    
    # For now, let's just do the removal phase.
    
    while active_q3s and time.time() - start_time < DEADLINE - 30:
        # Pick a random active Q3
        q3 = random.choice(list(active_q3s))
        # Pick a random edge from this Q3
        e = random.choice(list(q3))
        # Remove the edge
        edges.discard(e)
        # Deactivate all Q3s containing e
        for q in edge_to_q3s[e]:
            active_q3s.discard(q)
    
    print(f"After removal: {len(edges)} edges", file=sys.stderr)
    
    # Now, try to add edges back.
    # For each non-edge, check if adding it creates a Q3.
    # If not, add it.
    
    all_edges = set(frozenset((i, j)) for i in range(N) for j in range(i+1, N))
    non_edges = all_edges - edges
    
    # Shuffle non_edges
    non_edges = list(non_edges)
    random.shuffle(non_edges)
    
    for e in non_edges:
        if time.time() - start_time > DEADLINE - 10:
            break
        # Check if adding e creates any Q3
        creates_q3 = False
        for q3 in edge_to_q3s[e]:
            # Check if all other 11 edges of q3 are present
            if all(other in edges for other in q3 if other != e):
                creates_q3 = True
                break
        if not creates_q3:
            edges.add(e)
            # Activate Q3s that now have all edges
            for q3 in edge_to_q3s[e]:
                if all(other in edges for other in q3):
                    active_q3s.add(q3)
    
    print(f"After addition: {len(edges)} edges", file=sys.stderr)
    
    # Convert to list of [u, v] pairs
    result = [[u, v] for e in edges for u, v in [sorted(e)]]
    # Actually, frozenset doesn't preserve order. Let me fix.
    result = []
    for e in edges:
        u, v = tuple(e)
        if u > v:
            u, v = v, u
        result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END

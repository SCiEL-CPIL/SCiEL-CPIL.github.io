"""Dense Q3-free graph on 16 vertices via bitmask Q3 detection + local search."""
import time
import random
import itertools

# Deadline: evaluator kills at 150s; use 130s as safety margin
_DEADLINE = 130.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def construct():
    """Return a list of edges [u,v] of a Q3-free graph on vertices 0..15 maximizing edges."""
    N = 16
    # All possible edges
    all_edges = list(itertools.combinations(range(N), 2))
    n_edges_total = len(all_edges)  # 120

    # Precompute Q3 subgraph edge sets.
    # Q3 has 8 vertices and 12 edges.
    # The 3-cube Q3: vertices are 3-bit strings, edges connect vertices differing in one bit.
    # There are C(16,8) * (number of Q3 labelings on 8 vertices) possible Q3 subgraphs,
    # but many are isomorphic. Instead, we generate all distinct edge-sets of Q3 on 8 specific vertices.
    # Actually, for a given set of 8 vertices, there are multiple ways to embed Q3.
    # Better approach: precompute all Q3 edge-sets as frozensets of edge-pairs.

    # Q3 vertices: 0..7 as 3-bit numbers. Edges:
    # 0-1, 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
    q3_edges_canonical = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

    # Generate all distinct Q3 edge-sets on any 8-vertex subset of 0..15.
    # For each 8-vertex subset, try all 8! / 24 (automorphisms of Q3 is 48) permutations...
    # That's too many. Instead, for each 8-vertex subset, we check if the induced subgraph
    # contains a Q3. But for precomputation, let's think differently.

    # Alternative: precompute for every 8-vertex subset whether it can host a Q3,
    # and for each such subset, store the edge sets that form a Q3.
    # But that's C(16,8)=12870 subsets, each with 8! = 40320 permutations... too slow.

    # Better: use a different detection strategy.
    # For a given graph (as a set of edges), check if it contains Q3 by:
    # 1. For each 8-vertex subset, check if the induced subgraph is Q3 or contains Q3.
    # 2. But we need exact Q3 subgraph (not induced), so we check if there exists a
    #    bijection from Q3 vertices to the 8 vertices preserving edges.

    # Most efficient: precompute all Q3 edge-sets as frozensets.
    # Number of distinct Q3 subgraphs in K16: choose 8 vertices * number of non-isomorphic
    # Q3 embeddings. The automorphism group of Q3 has order 48 (hyperoctahedral group B3).
    # So for each 8-vertex set, there are 8!/48 = 840 distinct Q3 edge-sets.
    # Total: 12870 * 840 ≈ 10.8 million. That's a lot but manageable as frozensets?
    # Actually 10.8 million frozensets of 12 edges each would use too much memory.

    # Alternative approach: don't precompute all. Instead, use a fast detection function.
    # For local search, we add/remove edges and need to check if Q3 appears.
    # Fast check: when adding edge (u,v), check if this edge completes a Q3.
    # A Q3 containing edge (u,v): u and v are adjacent in Q3, so they differ in exactly 1 bit.
    # In Q3, each edge belongs to exactly 2 squares (4-cycles). So edge (u,v) is in exactly
    # 2 faces of Q3. The Q3 containing (u,v) is determined by... actually each edge is in
    # exactly 2 faces, and the Q3 is the union of specific faces.

    # Let me think about this more carefully. In Q3, each edge is in exactly 2 4-cycles (faces).
    # Given an edge (u,v) in Q3, the two faces containing it are determined. The remaining
    # 6 vertices of Q3 are determined by the two faces.

    # For detection: when we add edge (u,v), we need to check if there exist 6 other vertices
    # such that the 8-vertex induced subgraph (with all required Q3 edges present) forms a Q3.

    # Let's precompute for each edge (u,v) in K16, the set of 6-vertex sets S such that
    # {u,v} ∪ S forms a Q3 with (u,v) as an edge.

    # In Q3, label vertices by 3-bit strings. Edge (000, 001) [differ in bit 0].
    # The two faces containing this edge:
    # Face 1: 000-001-011-010-000 (bits 0 and 1 vary)
    # Face 2: 000-001-101-100-000 (bits 0 and 2 vary)
    # So the 6 other vertices are: 011, 010, 101, 100, 111, 110
    # Wait: 011, 010 from face 1; 101, 100 from face 2; and 111, 110 which are in both...
    # No. Let me redo. Q3 vertices: 000,001,010,011,100,101,110,111.
    # Edge (000,001). Faces containing this edge:
    #   Square with bits {0,1}: 000-001-011-010
    #   Square with bits {0,2}: 000-001-101-100
    # The remaining vertices: 110, 111.
    # So given edge (u,v) where u=000, v=001, the other 6 vertices are:
    #   010, 011, 100, 101, 110, 111
    # And the required edges are all 12 Q3 edges.

    # For general edge (u,v) in K16, we need to find all ways to assign 3-bit labels to
    # 8 vertices {u,v,w1,...,w6} such that (u,v) corresponds to an edge in Q3 and all
    # Q3 edges are present in the graph.

    # This is getting complex. Let me use a different strategy:
    # Precompute all Q3 edge-sets. There are C(16,8) * (8!/48) but many overlap.
    # Actually, let's just enumerate all 8-vertex subsets and for each, check all
    # permutations of Q3 vertices. But 12870 * 840 = 10.8M checks, each checking 12 edges.
    # That's ~130M edge checks, which should run in a few seconds in Python with optimization.

    # Actually, let me be smarter. For each 8-vertex subset, the number of Q3 embeddings
    # that are distinct as edge-sets is 8!/|Aut(Q3)| = 40320/48 = 840. But many of these
    # 840 will have the same edge-set? No, they're different edge-sets by construction.
    # Wait, two different bijections give the same edge-set iff they differ by an automorphism.
    # So yes, 840 distinct edge-sets per 8-vertex subset.

    # Total Q3 edge-sets: 12870 * 840 = 10,810,800. Storing each as a frozenset of 12 tuples
    # is too much memory. Let's store them as a set of frozensets... 10.8M frozensets won't fit.

    # Alternative: don't store all. Use a different detection method.
    # For the local search, when adding edge e, check if e is in any Q3 that is fully present.
    # We can precompute for each edge e, the list of Q3 edge-sets that contain e.
    # Each edge is in C(14,6) * (6!/48 * something)... let me compute.
    # Edge (u,v): choose 6 more vertices from remaining 14: C(14,6)=3003.
    # For each such 7-vertex set {u,v,w1..w6}, the number of Q3 embeddings with (u,v) as an edge:
    # In Q3, each edge has a specific structure. Given (u,v) as a specific edge type,
    # the number of ways to complete to a full Q3 labeling is:
    # After fixing u->000, v->001, the remaining 6 vertices must map to {010,011,100,101,110,111}.
    # The number of valid bijections: we need to assign these 6 labels to w1..w6 such that
    # all Q3 edges among the 8 are consistent. This is equivalent to the number of ways to
    # complete the partial labeling. By symmetry, this is 6! / |stabilizer|...
    # Actually, the number is 6! = 720 divided by the automorphisms that fix 000 and 001.
    # The stabilizer of edge (000,001) in Aut(Q3): automorphisms fixing 000 and 001.
    # These are permutations of the other 6 bits... Let me just say it's 6! / k for some k.
    # Empirically, for each 8-vertex set containing (u,v), the number of Q3 edge-sets
    # containing (u,v) is 840 / 12 = 70 (since each Q3 has 12 edges, and by symmetry each
    # edge appears in 840/12 = 70 of them).

    # So for each edge, there are 3003 * 70 = 210,210 Q3 edge-sets containing it.
    # Total: 120 * 210210 / 12 = 2,102,100... wait that doesn't match 10.8M.
    # 120 * 210210 = 25,225,200, divided by 12 (each Q3 counted 12 times) = 2,102,100.
    # But earlier I computed 10,810,800. Let me recheck.
    # C(16,8) = 12870. Per 8-vertex set, 8!/48 = 840. 12870*840 = 10,810,800.
    # Each Q3 has 12 edges, so total edge-Q3 incidences = 10,810,800 * 12 = 129,729,600.
    # Per edge: 129,729,600 / 120 = 1,081,080.
    # So each edge is in 1,081,080 Q3 edge-sets. That's a lot.

    # This precomputation approach is too memory-intensive. Let's use a different strategy.

    # Strategy: Use a fast Q3 detection function that works on the current graph.
    # For local search, we maintain the graph as a set of edges and a bitmask adjacency.
    # When adding an edge, we check if it creates a Q3.

    # Fast Q3 check when adding edge (u,v):
    # A Q3 containing (u,v) must have u and v as adjacent vertices in Q3.
    # In Q3, u and v are connected, so they differ in exactly 1 coordinate.
    # The two faces containing edge (u,v) in Q3 are 4-cycles.
    # Let's characterize: in Q3, edge (u,v) with u,v differing in bit i.
    # The two faces are: vary bits {i,j} and {i,k} for j≠i, k≠i, j≠k.
    # So the 6 other vertices are determined by the two "opposite" edges.
    #
    # For detection: given edge (u,v), find all pairs of vertices (a,b) and (c,d) such that
    # u-a-v-b-u and u-c-v-d-u are 4-cycles in the graph, and the remaining 2 vertices
    # complete the Q3.

    # This is still complex. Let me use a simpler but effective approach:
    # Use the fact that Q3 has girth 4 and is bipartite with parts of size 4.
    # A Q3 subgraph has 8 vertices, 12 edges, is 3-regular, bipartite (4+4), girth 4.

    # For practical purposes, let me implement a direct check:
    # Given the current edge set, to check if adding (u,v) creates a Q3:
    # 1. For each 6-vertex subset S of V \ {u,v}, check if {u,v} ∪ S contains a Q3
    #    with (u,v) as an edge.
    # 2. But C(14,6) = 3003 subsets, and for each we need to check ~70 Q3 embeddings.
    #    That's 210K checks per edge addition, each checking 12 edge presences.
    #    With ~100 edges to add, that's 21M * 12 = 252M operations. Too slow in Python.

    # Let me try a completely different approach:
    # Use a known dense Q3-free graph construction and then do local search.

    # Known: K_{3,13} has 39 edges and is Q3-free (Q3 needs 4+4 bipartition).
    # Can we do better? What about K_{4,12} minus some edges?
    # K_{4,12} contains Q3 (take 4 from each side that form Q3's bipartition).
    # So we need to remove edges from K_{4,12} to destroy all Q3 subgraphs.

    # Alternative: use a random dense graph and remove edges to destroy Q3s.
    # Start with K16 (120 edges), remove edges until Q3-free, trying to minimize removals.

    # Let me implement: start with K16, find all Q3 subgraphs, remove one edge from each
    # (greedily choosing the edge in the most Q3s), repeat until no Q3 remains.

    # But finding all Q3 subgraphs in K16 is the expensive part.

    # Let me try yet another approach: use the bipartite structure.
    # Q3 is bipartite with parts of size 4. So any Q3 subgraph has 4 vertices in one part
    # and 4 in the other, with all 12 edges between them forming Q3.

    # For a graph G, a Q3 subgraph exists iff there exist disjoint sets A, B of 4 vertices
    # each such that the bipartite subgraph between A and B is isomorphic to Q3's bipartite
    # subgraph (which is the complete bipartite K_{4,4} minus 4 edges... no, Q3 is 3-regular
    # so each vertex has degree 3, meaning in the bipartite view, it's a 3-regular bipartite
    # graph on 4+4 vertices, which is K_{4,4} minus a perfect matching).

    # So: G contains Q3 iff there exist 4+4 vertex sets A,B such that the bipartite
    # subgraph G[A,B] is 3-regular (each vertex has degree exactly 3 in the bipartite subgraph)
    # AND the bipartite graph is isomorphic to Q3 (i.e., not just any 3-regular bipartite
    # graph on 4+4, but specifically Q3).

    # Actually, all 3-regular bipartite graphs on 4+4 vertices are isomorphic to Q3?
    # No. There are multiple 3-regular bipartite graphs on 4+4. The number of 3-regular
    # bipartite graphs on 4+4: it's the number of 4x4 0-1 matrices with row/col sums 3,
    # which equals the number of perfect matchings in K_{4,4} complement...
    # A 4x4 0-1 matrix with all row/col sums 3 is the complement of a permutation matrix.
    # There are 4! = 24 such matrices. Up to isomorphism (permuting rows and columns),
    # how many distinct 3-regular bipartite graphs? The complement is a permutation matrix,
    # and permutation matrices up to row/col permutation: only one orbit (all permutation
    # matrices are isomorphic). So there's exactly one isomorphism class: Q3.
    #
    # So: G contains Q3 iff there exist 4+4 vertex sets A,B such that G[A,B] is 3-regular
    # bipartite (each vertex in A∪B has degree exactly 3 in G[A,B]).

    # This gives us a clean detection method!
    # For each pair of 4-sets (A,B) with A∩B=∅, check if G[A,B] is 3-regular bipartite.
    # Number of such pairs: C(16,4)*C(12,4) = 1820 * 495 = 900,900.
    # For each, check 8 degree conditions (4 in A, 4 in B). Very fast!

    # Let's implement this.

    def _contains_q3(adj_mask, edge_set):
        """Check if graph (given by adjacency bitmasks and edge set) contains Q3.
        adj_mask[v] is a bitmask of neighbors of v.
        Returns True if Q3 is present."""
        # For each 4-set A and 4-set B (disjoint), check if G[A,B] is 3-regular bipartite.
        vertices = list(range(16))
        for A in itertools.combinations(vertices, 4):
            A_set = set(A)
            remaining = [v for v in vertices if v not in A_set]
            for B in itertools.combinations(remaining, 4):
                # Check if G[A,B] is 3-regular bipartite
                ok = True
                for a in A:
                    # Count neighbors of a in B
                    cnt = 0
                    for b in B:
                        if adj_mask[a] & (1 << b):
                            cnt += 1
                    if cnt != 3:
                        ok = False
                        break
                if ok:
                    for b in B:
                        cnt = 0
                        for a in A:
                            if adj_mask[b] & (1 << a):
                                cnt += 1
                        if cnt != 3:
                            ok = False
                            break
                if ok:
                    return True
        return False

    def _contains_q3_fast(adj_mask):
        """Faster version using bit operations."""
        vertices = list(range(16))
        all_4sets = list(itertools.combinations(vertices, 4))
        for A in all_4sets:
            A_set = set(A)
            for B in all_4sets:
                if A_set & set(B):
                    continue
                # Check 3-regular bipartite between A and B
                ok = True
                for a in A:
                    mask_b = 0
                    for b in B:
                        mask_b |= (1 << b)
                    # Count bits in adj_mask[a] & mask_b
                    if bin(adj_mask[a] & mask_b).count('1') != 3:
                        ok = False
                        break
                if not ok:
                    continue
                for b in B:
                    mask_a = 0
                    for a in A:
                        mask_a |= (1 << a)
                    if bin(adj_mask[b] & mask_a).count('1') != 3:
                        ok = False
                        break
                if ok:
                    return True
        return False

    # Even faster: precompute all 4-sets and their masks
    all_4sets = list(itertools.combinations(range(16), 4))
    fourset_masks = [0] * len(all_4sets)
    for i, s in enumerate(all_4sets):
        m = 0
        for v in s:
            m |= (1 << v)
        fourset_masks[i] = m

    def _contains_q3_v2(adj_mask):
        """Check Q3 presence using precomputed 4-set masks."""
        n4 = len(all_4sets)
        for i in range(n4):
            A = all_4sets[i]
            maskA = fourset_masks[i]
            for j in range(n4):
                if maskA & fourset_masks[j]:
                    continue
                B = all_4sets[j]
                maskB = fourset_masks[j]
                ok = True
                for a in A:
                    if bin(adj_mask[a] & maskB).count('1') != 3:
                        ok = False
                        break
                if not ok:
                    continue
                for b in B:
                    if bin(adj_mask[b] & maskA).count('1') != 3:
                        ok = False
                        break
                if ok:
                    return True
        return False

    # Start with K16 and remove edges to destroy all Q3s
    # Actually, let's start with a denser initial guess and do local search.

    # Initialize: K_{3,13} = 39 edges. Let's try to do better.
    # Start with a random dense graph and remove edges.

    random.seed(42)

    def _random_dense_graph():
        """Generate a random graph with ~60% edge probability."""
        edges = []
        for u in range(16):
            for v in range(u+1, 16):
                if random.random() < 0.6:
                    edges.append((u, v))
        return set(edges)

    def _remove_edges_to_destroy_q3(edge_set):
        """Remove edges from edge_set until Q3-free. Returns (new_edge_set, num_removed)."""
        edge_set = set(edge_set)
        removed = 0
        while _time_left() > 5.0:
            # Build adjacency masks
            adj_mask = [0] * 16
            for u, v in edge_set:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)

            if not _contains_q3_v2(adj_mask):
                break

            # Find a Q3 and remove an edge from it
            # We need to find an actual Q3, not just detect existence
            # Let's find one and remove a random edge from it
            vertices = list(range(16))
            found = False
            for i in range(len(all_4sets)):
                if found:
                    break
                A = all_4sets[i]
                maskA = fourset_masks[i]
                for j in range(len(all_4sets)):
                    if maskA & fourset_masks[j]:
                        continue
                    B = all_4sets[j]
                    maskB = fourset_masks[j]
                    ok = True
                    for a in A:
                        if bin(adj_mask[a] & maskB).count('1') != 3:
                            ok = False
                            break
                    if not ok:
                        continue
                    for b in B:
                        if bin(adj_mask[b] & maskA).count('1') != 3:
                            ok = False
                            break
                    if ok:
                        # Found Q3 on A ∪ B
                        # Collect all edges in this Q3
                        q3_edges = []
                        for a in A:
                            for b in B:
                                if adj_mask[a] & (1 << b):
                                    q3_edges.append((a, b))
                        # Remove a random edge
                        e = random.choice(q3_edges)
                        edge_set.discard(e)
                        removed += 1
                        found = True
                        break
        return edge_set, removed

    def _try_add_edges(edge_set):
        """Try to add edges back while maintaining Q3-freeness. Returns (new_edge_set, num_added)."""
        edge_set = set(edge_set)
        added = 0
        # Build list of missing edges
        all_edges = list(itertools.combinations(range(16), 2))
        present = set(edge_set)
        missing = [e for e in all_edges if e not in present]
        random.shuffle(missing)

        for u, v in missing:
            if _time_left() < 3.0:
                break
            # Try adding (u, v)
            new_set = edge_set | {(u, v)}
            adj_mask = [0] * 16
            for a, b in new_set:
                adj_mask[a] |= (1 << b)
                adj_mask[b] |= (1 << a)
            if not _contains_q3_v2(adj_mask):
                edge_set = new_set
                added += 1

        return edge_set, added

    # Main search: multiple restarts
    best_edges = set()
    best_count = 0

    # Start with K_{3,13}
    k313 = set()
    for a in range(3):
        for b in range(3, 16):
            k313.add((a, b))
    best_edges = set(k313)
    best_count = len(k313)

    # Try random restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if _time_left() < 10.0:
            break

        # Generate random dense graph
        density = 0.5 + 0.3 * (restart / num_restarts)
        edge_set = set()
        for u in range(16):
            for v in range(u+1, 16):
                if random.random() < density:
                    edge_set.add((u, v))

        # Remove edges to destroy Q3s
        edge_set, _ = _remove_edges_to_destroy_q3(edge_set)

        # Try to add edges back
        edge_set, _ = _try_add_edges(edge_set)

        # Local search: try swapping edges
        if _time_left() > 15.0:
            for _ in range(20):
                if _time_left() < 10.0:
                    break
                # Try removing a random edge and adding two others
                if not edge_set:
                    break
                rem_edge = random.choice(list(edge_set))
                new_set = edge_set - {rem_edge}
                # Try to add up to 2 edges
                all_edges = list(itertools.combinations(range(16), 2))
                present = set(new_set)
                missing = [e for e in all_edges if e not in present]
                random.shuffle(missing)
                added_any = False
                for e in missing[:5]:
                    candidate = new_set | {e}
                    adj_mask = [0] * 16
                    for a, b in candidate:
                        adj_mask[a] |= (1 << b)
                        adj_mask[b] |= (1 << a)
                    if not _contains_q3_v2(adj_mask):
                        new_set = candidate
                        added_any = True
                        break
                if len(new_set) > len(edge_set):
                    edge_set = new_set

        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = set(edge_set)

    # Final: try to add more edges to best
    if _time_left() > 5.0:
        best_edges, _ = _try_add_edges(best_edges)

    return [[u, v] for u, v in best_edges]

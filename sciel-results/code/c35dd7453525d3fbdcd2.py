"""Kissing number in dimension 11 - MIS on conflict graph of norm-4 vectors."""
import numpy as np
import time
import sys
from itertools import combinations, product

D = 11
MAX_TIME = 140.0  # Safety margin before 150s kill


def construct():
    start_time = time.time()

    # Generate all integer vectors in R^11 with exactly 4 nonzero entries, each ±1.
    # Count: C(11,4) * 2^4 = 330 * 16 = 5280 vectors.
    # Each has norm sqrt(4) = 2.
    #
    # The kissing condition: min pairwise distance >= max norm.
    # If all vectors have norm 2, we need min pairwise distance >= 2.
    # ||u - v|| >= 2 for all pairs.
    # ||u - v||^2 = 4 + 4 - 2*u.v = 8 - 2*u.v >= 4
    # u.v <= 2
    #
    # For two weight-4 vectors, u.v can be:
    # - If they share k coordinates: u.v = sum of products over shared coords
    #   For each shared coordinate, product is ±1
    #   So u.v ranges from -k to k in steps of 2
    # - If they share 0 coordinates: u.v = 0
    # - If they share 1 coordinate: u.v = ±1
    # - If they share 2 coordinates: u.v = -2, 0, or 2
    # - If they share 3 coordinates: u.v = -3, -1, 1, or 3
    # - If they share 4 coordinates: u.v = -4, -2, 0, 2, or 4
    #
    # Conflict: u.v > 2, i.e., u.v >= 3
    # This happens when they share 3 or 4 coordinates with aligned signs.
    #
    # We want the largest set of vectors with no conflicts (independent set).

    # Generate all vectors
    vectors = []
    for combo in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)

    n = len(vectors)  # 5280
    vectors = np.array(vectors)  # shape (n, D)

    # Build conflict graph as adjacency matrix (or sparse representation)
    # Two vectors conflict if their dot product > 2
    # Compute all pairwise dot products
    # dot[i,j] = vectors[i] . vectors[j]
    # Conflict if dot[i,j] > 2 (and i != j)

    # For memory efficiency with 5280 vectors, the full matrix is 5280*5280*8 = ~223MB
    # Let's use a more memory-efficient approach
    # Actually, let's compute the conflict structure using bitsets

    # Compute dot products in batches to save memory
    # Actually, 5280^2 * 8 bytes = 223 MB, which is fine
    dots = vectors @ vectors.T  # (n, n) matrix of dot products

    # Conflict graph: edge between i and j if dots[i,j] > 2
    # Note: dots[i,i] = 4 (norm squared), so diagonal is always > 2 but we exclude self
    conflict = (dots > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Now solve Maximum Independent Set on this graph
    # Equivalent to Maximum Clique on the complement graph
    # Or use branch-and-bound directly

    # Use a branch-and-bound approach for MIS
    # Key idea:
    # 1. Order vertices by degree (or use a greedy coloring of the complement for upper bound)
    # 2. Branch: include or exclude a vertex
    # 3. Upper bound: use greedy coloring of the remaining candidate subgraph

    # For the upper bound, we need to color the conflict graph (not complement)
    # Wait - for MIS, the upper bound comes from coloring the conflict graph.
    # If we can color the conflict graph with c colors, then the MIS has size at most c.
    # (Because in an independent set, no two vertices share a color in the conflict graph coloring)

    # Actually, let me reconsider. For Maximum Independent Set:
    # - The conflict graph G has edges between conflicting vertices
    # - An independent set in G is a set of vertices with no edges between them
    # - Upper bound: if we color G with c colors, the independent set has size at most c
    #   (each color class is an independent set in the complement, but we need independent in G)
    #   Wait, that's not right. Let me think again.
    #
    # For MIS in graph G:
    # - Greedy coloring of G gives an upper bound on the size of the maximum clique in G
    # - For MIS, we can use the greedy coloring of the COMPLEMENT of G
    # - Actually, the standard approach: the upper bound for MIS is the chromatic number of G
    #   No wait...
    #
    # Let me use the standard Bron-Kerbosch style or the standard MIS branch-and-bound:
    # Upper bound for the number of vertices we can still add = size of maximum independent set
    # in the remaining subgraph. A simple upper bound: greedy coloring of the remaining
    # conflict subgraph. If we can color it with c colors, the MIS is at most c.
    #
    # Actually the correct statement: if we color the conflict graph G with c colors,
    # then any independent set in G has at most one vertex per color class? No, that's wrong.
    # An independent set in G can have multiple vertices of the same color (since same color
    # means no edge in G, which is exactly what we want for independent set).
    #
    # I had it backwards. Let me use the correct approach:
    # For MIS in G, the upper bound is obtained by coloring the COMPLEMENT of G.
    # If complement of G can be colored with c colors, then MIS in G has size at most c.
    # Because: independent set in G = clique in complement of G.
    # And a clique can have at most one vertex per color in any proper coloring.

    # Hmm, this is getting complicated. Let me use a simpler but effective approach:
    # Greedy heuristic with local search (swap-based improvement).

    # Actually, let me try a different structural approach:
    # Use a greedy algorithm with smart ordering, then improve with local search.

    def greedy_mis(conflict, order=None):
        """Greedy MIS: process vertices in given order, add if no conflict with selected."""
        n = len(conflict)
        if order is None:
            order = list(range(n))
        selected = set()
        for v in order:
            # Check if v conflicts with any selected vertex
            conflict_with_selected = False
            for s in selected:
                if conflict[v, s]:
                    conflict_with_selected = True
                    break
            if not conflict_with_selected:
                selected.add(v)
        return selected

    # Better approach: use vectorized operations
    def greedy_mis_fast(conflict, order):
        """Vectorized greedy MIS."""
        n = len(conflict)
        selected = np.zeros(n, dtype=bool)
        for v in order:
            if not np.any(conflict[v] & selected):
                selected[v] = True
        return selected

    # Even faster: process in order, maintain a "blocked" set
    def greedy_mis_v2(conflict, order):
        n = len(conflict)
        selected = []
        blocked = np.zeros(n, dtype=bool)
        for v in order:
            if not blocked[v]:
                selected.append(v)
                blocked |= conflict[v]
        return selected

    # Try multiple orderings
    # Ordering 1: by degree (ascending - prefer low degree)
    degrees = conflict.sum(axis=1)
    order_by_degree = np.argsort(degades) if False else np.argsort(degrees)

    # Ordering 2: random
    rng = np.random.RandomState(42)
    order_random = rng.permutation(n)

    # Ordering 3: by degree (descending)
    order_by_degree_desc = np.argsort(-degrees)

    # Try greedy with different orderings
    best_set = None
    best_size = 0

    for order in [order_by_degree, order_random, order_by_degree_desc]:
        if time.time() - start_time > MAX_TIME - 30:
            break
        selected = greedy_mis_v2(conflict, order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected

    # Local search: try to swap out a vertex for two that don't conflict
    # or add a vertex by removing one that conflicts with it
    if best_set is not None:
        selected_arr = np.zeros(n, dtype=bool)
        selected_arr[best_set] = True

        # Try adding vertices one at a time by removing conflicting ones
        # For each non-selected vertex, check how many selected vertices it conflicts with
        # If it conflicts with exactly 1 selected vertex, we can swap (remove 1, add 1 = net 0)
        # If it conflicts with 0, we can add (net +1) - but greedy should have caught this
        # Actually, due to ordering, some might have been blocked but not selected

        # Try: for each non-selected vertex v, if it conflicts with k selected vertices,
        # we could remove those k and add v. Net change: 1 - k.
        # This is only beneficial if k = 0 (which greedy handles) or we find a better swap.

        # Better local search: try pairs
        # For each pair of non-selected vertices (v1, v2) that don't conflict with each other,
        # check if removing their conflicts from selected and adding v1, v2 improves.

        # This is expensive. Let's do a simpler 1-swap:
        # For each non-selected v, count conflicts with selected
        # If conflicts == 0: add v (shouldn't happen with good greedy, but check)
        # If conflicts == 1: try the swap (remove the 1, add v) - net 0, but might enable future adds
        # Skip pure swaps for now, focus on the structure.

        # Try multiple random restarts of greedy
        for restart in range(5):
            if time.time() - start_time > MAX_TIME - 20:
                break
            order_r = rng.permutation(n)
            selected_r = greedy_mis_v2(conflict, order_r)
            if len(selected_r) > best_size:
                best_size = len(selected_r)
                best_set = selected_r

    # Now try a branch-and-bound on the most promising subset
    # Focus on the vertices that are in the best_set or near it
    # Actually, let's try a smarter approach:
    # Use the structure of the problem. Vectors are weight-4 with ±1 entries.
    # Two vectors conflict if their dot product > 2.
    #
    # Let's try a different decomposition:
    # Group vectors by their support (set of 4 coordinates).
    # For a fixed support S of size 4, there are 16 vectors (2^4 sign patterns).
    # Two vectors with the same support S: their dot product is the number of matching signs minus
    # the number of mismatching signs = 4 - 2*(Hamming distance in signs)
    # So dot = 4, 2, 0, -2, -4 for Hamming distance 0, 1, 2, 3, 4.
    # Conflict if dot > 2, i.e., dot = 4 (same vector) or... wait, dot = 4 only for identical.
    # For distinct vectors with same support: dot can be 2, 0, -2, -4. None > 2.
    # So NO two distinct vectors with the same support conflict!
    #
    # Wait, that means for each support, we can include ALL 16 vectors!
    # Let me verify: two vectors with same support {i,j,k,l}:
    # v1 = (s1, s2, s3, s4) at positions (i,j,k,l)
    # v2 = (t1, t2, t3, t4) at positions (i,j,k,l)
    # dot = s1*t1 + s2*t2 + s3*t3 + s4*t4
    # Each term is ±1, so dot ranges from -4 to 4 in steps of 2.
    # For distinct vectors, dot < 4. So dot <= 2. No conflict!
    #
    # Great! So within each support group, all 16 vectors are mutually non-conflicting.
    #
    # Now, vectors with different supports:
    # Support S1 and S2, |S1 ∩ S2| = k (k = 0, 1, 2, 3)
    # (k=4 means same support, handled above)
    # dot = sum over S1∩S2 of (sign products)
    # If k=0: dot = 0, no conflict
    # If k=1: dot = ±1, no conflict
    # If k=2: dot = -2, 0, or 2, no conflict (2 is not > 2)
    # If k=3: dot = -3, -1, 1, or 3. Conflict if dot = 3!
    #
    # So conflicts only occur between vectors whose supports intersect in exactly 3 coordinates,
    # and where the 3 shared coordinates have matching signs (giving dot = 3).
    #
    # This is a much simpler structure! Let me exploit this.
    #
    # For each pair of supports S1, S2 with |S1 ∩ S2| = 3:
    # Let T = S1 ∩ S2 (size 3), and a = S1 \ S2, b = S2 \ S1 (each size 1)
    # v1 has signs (s_T, s_a) where s_T is a vector in {±1}^3 and s_a = ±1
    # v2 has signs (t_T, t_b) where t_T is a vector in {±1}^3 and t_b = ±1
    # dot(v1, v2) = s_T · t_T (the 3 shared coordinates)
    # Conflict if s_T · t_T = 3, i.e., s_T = t_T (all 3 signs match)
    #
    # So: v1 and v2 conflict iff they share 3 coordinates AND have the same signs on those 3 coordinates.
    #
    # This means: for each triple T of coordinates, and each sign pattern on T (8 patterns),
    # all vectors that extend this (T, pattern) to a 4-set conflict with each other.
    #
    # Let me restructure:
    # A vector is determined by (support S of size 4, sign pattern on S).
    # Equivalently, by choosing 3 coordinates T and a 4th coordinate a, with signs.
    #
    # Conflict: two vectors conflict iff they share exactly 3 coordinates with the same signs.
    #
    # Let me think of it as: for each "triple-sign" (T, s_T) where T is a 3-subset and s_T is a sign pattern on T,
    # the vectors that "use" this triple-sign are those with support containing T and signs on T equal to s_T.
    # For a given (T, s_T), the vectors using it are: for each a in [D]\T, the vector with support T∪{a} and signs (s_T, s_a) for s_a ∈ {±1}.
    # So there are 2*(D-3) = 2*8 = 16 vectors using a given (T, s_T).
    # All 16 of these vectors conflict with each other (they share T with signs s_T).
    #
    # So the conflict graph has a structure based on these "stars" centered at (T, s_T).
    #
    # Number of (T, s_T) pairs: C(11,3) * 8 = 165 * 8 = 1320.
    # Each has 16 vectors in its star.
    # Each vector belongs to exactly 4 such stars (one for each of its 4 triples).
    #
    # This is a 4-uniform hypergraph structure. The problem is: select the maximum number of vectors
    # such that no (T, s_T) star has 2 or more selected vectors.
    #
    # This is equivalent to: for each of the 1320 stars, select at most 1 vector from it.
    # Each vector is in 4 stars.
    #
    # This is a packing problem! We want to pack as many vectors as possible with the constraint
    # that each star contains at most 1 selected vector.
    #
    # This is like a 4-uniform hypergraph matching/packing problem, but with the constraint
    # being on the "edges" (stars) rather than vertices.
    #
    # Actually, it's the dual: we have 1320 "bins" (stars) and 5280 "items" (vectors),
    # each item goes into 4 bins, and we can select an item only if all 4 of its bins are empty.
    #
    # This is equivalent to finding a maximum set of vectors where no two share a (T, s_T).
    #
    # Let me reformulate: assign to each vector a unique "signature" of 4 triple-signs.
    # Two vectors conflict iff they share at least one triple-sign.
    # We want the maximum set of vectors with pairwise disjoint triple-sign sets.
    #
    # This is a maximum packing in a 4-uniform hypergraph where edges are the triple-signs
    # and we're selecting hyperedges (vectors) that are edge-disjoint.
    # Wait no, it's the opposite. Let me re-examine.
    #
    # Stars = (T, s_T), there are 1320 of them.
    # Vectors = 5280 of them, each belongs to 4 stars.
    # Constraint: at most 1 vector per star.
    # Goal: maximize number of selected vectors.
    #
    # This is a maximum independent set in the conflict graph, which we already established.
    # But the structure helps: it's a "4-regular" incidence between 5280 vectors and 1320 stars.
    #
    # Greedy approach: process stars, for each star pick at most 1 vector from it.
    # But vectors span multiple stars, so we need to be careful.
    #
    # Alternative: this is equivalent to a maximum matching in a bipartite graph?
    # No, it's more like a set packing problem.
    #
    # Let me just use the greedy + local search approach, but with the structural insight
    # to speed up conflict checking.

    # Actually, let me just use the direct approach with the conflict matrix we already have.
    # The key insight is that the problem is well-structured, so greedy should work well.

    # Let me try a more sophisticated greedy: process in order of "constraint tightness"
    # A vector that conflicts with many others should be processed later (to avoid blocking many options)
    # Or: process low-degree vertices first (they block fewer options)

    # Let me also try: since within each support all 16 vectors are compatible,
    # we should try to include as many full supports as possible.

    # New strategy:
    # 1. For each support (330 of them), all 16 vectors are mutually compatible.
    # 2. Two vectors from different supports conflict only if their supports share 3 coords with matching signs.
    # 3. Greedy: process supports one by one. For each support, decide which (if any) of its 16 vectors to include.
    #    Actually, we can include ALL 16 from a support as long as they don't conflict with previously selected.

    # Let me try: select all 16 vectors from as many supports as possible.
    # Two supports S1, S2 conflict if they share 3 coordinates.
    # C(11,4) = 330 supports. Two supports share 3 coords if they differ in exactly 1 element.
    # Number of such pairs: for each support S, there are (11-4)*4 = 28 other supports sharing 3 coords.
    #
    # But the conflict also depends on signs! Two vectors from supports S1, S2 (sharing 3 coords T)
    # conflict only if their signs on T match.
    #
    # So if we include all 16 vectors from support S1, they conflict with vectors in support S2
    # only for the 8 sign patterns on T that match. Specifically, for each of the 8 sign patterns on T,
    # 2 vectors in S1 (varying the 4th sign) conflict with 2 vectors in S2 (varying the 4th sign).
    # So 8*2 = 16 pairs of conflicting vectors between the two groups of 16.
    #
    # This means we can't just take all 16 from adjacent supports.
    #
    # Let me go back to the direct MIS approach with the conflict matrix.

    # Use a better branch-and-bound:
    # 1. Start with the greedy solution
    # 2. Try to improve by:
    #    a. Removing a vertex and trying to add 2+ vertices in its neighborhood
    #    b. Simulated annealing on the selection

    # Let me try a different approach: use scipy's MILP or just a very good greedy

    # Actually, let me try the following efficient approach:
    # Since the conflict structure is well-understood, let's use a color-coding /
    # maximum matching approach on the star structure.

    # For now, let's use multiple greedy passes with different orderings and take the best.

    # Generate more diverse orderings
    best_set_final = best_set
    best_size_final = best_size

    # Try ordering by "star degree" - how many stars a vector is in (always 4)
    # Try ordering by the number of conflicts within its support group (always 0)
    # Try ordering by the minimum number of conflicts with vectors in each adjacent support

    # Let's try: process supports in order, and for each support, include vectors
    # that don't conflict with already-selected vectors.

    # Map vectors to their supports
    # Vector index -> support (frozenset of 4 coordinates)
    support_list = []
    for combo in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            support_list.append(combo)

    # Group vectors by support
    from collections import defaultdict
    support_to_vectors = defaultdict(list)
    for idx, sup in enumerate(support_list):
        support_to_vectors[sup].append(idx)

    # Process supports in order of size (all same size) but try different orderings
    supports = list(support_to_vectors.keys())

    for trial in range(10):
        if time.time() - start_time > MAX_TIME - 10:
            break
        if trial == 0:
            sup_order = supports
        else:
            sup_order = supports.copy()
            rng.shuffle(sup_order)

        selected = np.zeros(n, dtype=bool)
        for sup in sup_order:
            vecs = support_to_vectors[sup]
            # Try to add each vector in this support
            for v in vecs:
                if not np.any(conflict[v] & selected):
                    selected[v] = True

        if selected.sum() > best_size_final:
            best_size_final = int(selected.sum())
            best_set_final = np.where(selected)[0]

    # Convert to final answer
    if best_set_final is not None:
        result = vectors[best_set_final].astype(np.int64)
    else:
        # Fallback: just use the first 440 weight-2 vectors
        result = np.zeros((1, D), dtype=np.int64)

    # Verify
    if len(result) > 0:
        norms = np.sqrt(np.sum(result**2, axis=1))
        max_norm = np.max(norms)
        # Check pairwise distances
        valid = True
        for i in range(min(len(result), 200)):  # Spot check
            for j in range(i+1, min(len(result), 200)):
                d = np.sqrt(np.sum((result[i] - result[j])**2))
                if d < max_norm - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fall back to seed
            seed_vecs = []
            for i in range(D):
                for j in range(i+1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            seed_vecs.append(v)
            result = np.array(seed_vecs, dtype=np.int64)

    return result

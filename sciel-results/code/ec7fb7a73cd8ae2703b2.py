import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build conflict graph: conflict if intersection size >= 3
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute popcount for 11-bit integers
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Vectorized conflict matrix
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = (popcounts >= 3)
    np.fill_diagonal(conflict, False)

    # Convert conflict to integer adjacency for vectorized operations
    adj = conflict.astype(np.int8)

    # Maximum Independent Set via Branch and Bound with greedy coloring
    # We want the largest set of non-conflicting supports
    # This is equivalent to Maximum Independent Set on the conflict graph
    # Or Maximum Clique on the complement graph (non-conflict graph)

    # Use branch and bound on the conflict graph for maximum independent set
    # A node in the independent set can be added if it doesn't conflict with existing members

    # Greedy coloring for upper bound on independent set size
    # For maximum independent set, we can use greedy coloring on the graph
    # The chromatic number gives an upper bound on clique size, but for independent set
    # we need a different approach. Let's use the fact that if we color the graph,
    # each color class is an independent set, so the maximum independent set size
    # is at most the number of vertices (trivial) but we can use greedy heuristics.

    # Actually, for maximum independent set, a good upper bound is:
    # |S| where S is the candidate set, and we can use greedy coloring to bound
    # Let's use a different approach: branch and bound with a simple upper bound

    def greedy_color_bound(candidates):
        """
        Compute an upper bound on the maximum independent set size
        within the candidate set using a greedy approach.
        """
        n = len(candidates)
        if n == 0:
            return 0
        if n <= 8:
            return n  # Small enough, just return n

        # Greedy coloring: assign colors to vertices
        # Each color class is an independent set
        # The number of colors used is an upper bound on the independence number?
        # No, that's not right. Let me think again.

        # For maximum independent set, a simple upper bound is n.
        # A better bound: use the fact that in any graph, the independence number
        # is at most n - minimum vertex cover, and minimum vertex cover >= maximum matching.
        # But that's complex. Let's just use a simple greedy bound.

        # Actually, let me use a different strategy:
        # The upper bound for the remaining search is simply the number of candidates.
        # To make it tighter, we can use a greedy packing:
        # Greedily pick vertices that don't conflict, count them, that's a lower bound.
        # For upper bound, we can use: for each vertex v, the max independent set
        # containing v has size at most 1 + (max independent set in N(v) complement).

        # Let's use a simple but effective bound:
        # Sort candidates by degree (descending), greedily pick non-conflicting ones
        # This gives a lower bound. For upper bound, use n.
        # To get a tighter upper bound, use the fact that:
        # alpha(G) <= n - tau(G) where tau is minimum vertex cover
        # And tau(G) >= matching_number(G)
        # But computing matching is expensive.

        # For now, use a simple bound: the number of candidates
        # But we can improve by noting that if we have a clique in the conflict graph,
        # we can pick at most 1 from it.

        # Let's use a greedy coloring approach for the complement:
        # In the complement graph (non-conflict), we want maximum clique.
        # Greedy coloring on the complement gives an upper bound on clique size.

        # Build subgraph on candidates
        cand_set = set(candidates)
        cand_list = list(candidates)
        n_cand = len(cand_list)

        # For each vertex, compute its non-neighbors within candidates
        # (i.e., vertices that don't conflict with it)
        # A clique in the non-conflict graph = independent set in conflict graph

        # Greedy coloring on the non-conflict subgraph
        # Color assignment: vertex v gets color c if no neighbor (in non-conflict graph) has color c
        # The number of colors is an upper bound on the clique size

        # Non-conflict adjacency within candidates
        # Two vertices are adjacent in non-conflict if they don't conflict
        # i.e., NOT conflict[i,j]

        # For greedy coloring, we process vertices in some order
        # and assign the smallest available color

        # Order by degree in conflict graph (descending) - high degree first
        degrees = []
        for v in cand_list:
            # Count conflicts within candidates
            d = 0
            for u in cand_list:
                if u != v and conflict[v, u]:
                    d += 1
            degrees.append(d)

        # Sort by degree descending
        order = sorted(range(n_cand), key=lambda i: -degrees[i])

        # Assign colors
        color = [-1] * n_cand
        next_color = 0

        for idx in order:
            v = cand_list[idx]
            # Find colors used by neighbors in non-conflict graph
            # Neighbors in non-conflict = vertices that don't conflict with v
            used_colors = set()
            for jdx in range(n_cand):
                if jdx == idx:
                    continue
                u = cand_list[jdx]
                if not conflict[v, u]:  # non-conflict, so adjacent in non-conflict graph
                    if color[jdx] != -1:
                        used_colors.add(color[jdx])

            # Find smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color[idx] = c
            if c + 1 > next_color:
                next_color = c + 1

        return next_color

    # Branch and bound for maximum independent set
    best_size = 0
    best_set = set()

    # Initial greedy solution
    def greedy_is():
        """Greedy maximum independent set"""
        remaining = set(range(N_S))
        selected = set()
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and conflict[v, u])
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            if min_v == -1:
                break
            selected.add(min_v)
            remaining.discard(min_v)
            # Remove all neighbors of min_v
            to_remove = set()
            for u in remaining:
                if conflict[min_v, u]:
                    to_remove.add(u)
            remaining -= to_remove
        return selected

    # Get initial solution
    init_set = greedy_is()
    best_size = len(init_set)
    best_set = set(init_set)

    # Branch and bound
    # State: (candidates, current_size, current_set)
    # candidates: list of vertices that can still be added
    # We branch on a vertex: include it or exclude it

    import sys
    sys.setrecursionlimit(10000)

    def branch_and_bound(candidates, current_size, current_set):
        nonlocal best_size, best_set

        if time.time() > deadline - 5:
            return

        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return

        # Upper bound: current_size + number of candidates
        # Tighter: current_size + greedy_color_bound(candidates)
        ub = current_size + greedy_color_bound(candidates)
        if ub <= best_size:
            return

        # Pick a vertex to branch on
        # Choose vertex with maximum degree in the candidate subgraph
        v = candidates[0]
        max_deg = -1
        for c in candidates:
            deg = sum(1 for u in candidates if u != c and conflict[c, u])
            if deg > max_deg:
                max_deg = deg
                v = c

        # Branch 1: Include v
        new_candidates = [c for c in candidates if c != v and not conflict[v, c]]
        new_set = set(current_set)
        new_set.add(v)
        branch_and_bound(new_candidates, current_size + 1, new_set)

        # Branch 2: Exclude v
        remaining = [c for c in candidates if c != v]
        branch_and_bound(remaining, current_size, current_set)

    # Start with all vertices as candidates
    all_candidates = list(range(N_S))
    branch_and_bound(all_candidates, 0, set())

    # Use the best set found
    best_support_set = best_set

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # Greedy fill: try to add more Type D vectors from non-selected supports
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    rng = np.random.RandomState(42)
    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 5:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue
        can_add = True
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot >= 3:
                can_add = False
                break
        if can_add:
            current_set_vectors.append(v)
            current_set_vectors_set.add(tuple(v))

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors

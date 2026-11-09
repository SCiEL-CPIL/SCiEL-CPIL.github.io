import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))

        # Bitmask for clique edges
        edge_mask = [0] * k
        for u in range(k):
            for v in range(u + 1, k):
                edge_mask[u] |= (1 << v)
                edge_mask[v] |= (1 << u)

        partition = []

        # Process independent set vertices
        # Sort by neighborhood size ascending
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)))

        for v in I_sorted:
            N_v = adj[v] & set(C)

            if not N_v:
                continue

            # Group independent vertices with same neighborhood
            # Build a mapping from neighborhood mask to list of vertices
            # Actually, we need to group by the SET of neighbors in C
            # Let's use a dict: frozenset(neighborhood) -> [vertices]
            # But we need to be careful: we want to group vertices that have
            # IDENTICAL neighborhoods in C, so that a single clique can cover
            # all their edges to C at once.

            # First, let's collect all remaining independent vertices with their neighborhoods
            # We'll process them in groups

            # Build groups: key = sorted tuple of neighbors, value = list of vertices
            # But we need to process in a specific order. Let's collect all I vertices
            # that haven't been processed yet.

            # Actually, let's restructure: process all I vertices, grouping by neighborhood
            # This is a significant change. Let me think about this differently.

            # For now, let's keep the per-vertex approach but improve the clique finding.
            # The key insight from the strategy: group independent vertices with identical
            # or nested neighborhoods.

            # Let's try a different approach:
            # 1. For each independent vertex, compute its neighborhood mask in C
            # 2. Group vertices by identical neighborhood
            # 3. For each group, the shadow is the intersection of neighborhoods (same for identical)
            # 4. Cover the shadow with cliques, and for each clique, create one clique per vertex
            #    in the group? No, that doesn't work because the clique must be a clique in G.

            # Actually, if vertices v1, v2 have the same neighborhood N in C, then {v1, v2} is NOT
            # an edge (they're in the independent set). So we can't put them in the same clique.
            # But we CAN use the same clique in C to cover edges from both v1 and v2.
            # So for each clique S in C, we create cliques {v1} ∪ S and {v2} ∪ S.
            # This doesn't reduce the number of cliques for the cross edges, but it might
            # help with the C-part edges.

            # Wait, the C-part edges are separate. The cross edges from v1 to S and v2 to S
            # are covered by two separate cliques. So grouping doesn't help reduce cross-edge
            # cliques.

            # Let me reconsider. The current approach processes each I vertex independently.
            # The number of cliques for cross edges is at least the number of I vertices with
            # non-empty neighborhoods, since each needs at least one clique.

            # So the key optimization is in the C-part: minimize the number of cliques needed
            # to cover the remaining edges in C after removing edges covered by cross-edge cliques.

            # Let's focus on making the C-part clique cover more efficient.

            # For each I vertex, find a maximal clique in N_v (induced subgraph on N_v in G[C])
            # This removes more edges from C, leaving fewer edges to cover separately.

            while N_v:
                # Find a maximal clique in the induced subgraph on N_v
                # Use a greedy algorithm with bitmask operations

                # Start with vertex of max degree in N_v
                nv_mask = 0
                for u in N_v:
                    nv_mask |= (1 << u)

                # Find vertex with max degree in N_v
                max_deg_vertex = None
                max_deg = -1
                for u in N_v:
                    deg = bin(edge_mask[u] & nv_mask).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        max_deg_vertex = u

                current_clique = {max_deg_vertex}
                common_mask = edge_mask[max_deg_vertex] & nv_mask

                # Greedily extend the clique
                while True:
                    best_candidate = None
                    best_score = -1

                    temp = common_mask
                    while temp:
                        lsb = temp & (-temp)
                        w = lsb.bit_length() - 1
                        temp ^= lsb

                        # Score: degree in remaining common neighborhood
                        score = bin(edge_mask[w] & common_mask).count('1')
                        if score > best_score:
                            best_score = score
                            best_candidate = w

                    if best_candidate is None:
                        break

                    current_clique.add(best_candidate)
                    common_mask &= edge_mask[best_candidate]

                # Add clique to partition
                clique_list = sorted(list(current_clique | {v}))
                partition.append(clique_list)

                # Remove edges of current_clique from edge_mask
                clique_list_sorted = sorted(current_clique)
                for i in range(len(clique_list_sorted)):
                    u = clique_list_sorted[i]
                    for j in range(i + 1, len(clique_list_sorted)):
                        w = clique_list_sorted[j]
                        edge_mask[u] &= ~(1 << w)
                        edge_mask[w] &= ~(1 << u)

                # Remove current_clique from N_v
                N_v -= current_clique

        # Remaining edges in C: partition into cliques
        # Use a more efficient algorithm: find all maximal cliques and use a greedy set cover

        # First, find all remaining edges
        remaining_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if edge_mask[u] & (1 << v):
                    remaining_edges.append((u, v))

        # If no remaining edges, we're done
        if not remaining_edges:
            return partition

        # Build adjacency for remaining edges
        rem_adj = [set() for _ in range(k)]
        for u, v in remaining_edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)

        # Use a greedy algorithm: repeatedly find a maximal clique in the remaining graph
        # and add it to the partition

        while remaining_edges:
            # Pick an edge with maximum common neighborhood
            best_edge = None
            best_common = -1

            for e in remaining_edges:
                u, v = e
                common_mask = edge_mask[u] & edge_mask[v]
                common_count = bin(common_mask).count('1')
                if common_count > best_common:
                    best_common = common_count
                    best_edge = e

            if best_edge is None:
                best_edge = remaining_edges[0]

            u, v = best_edge
            current_clique = {u, v}

            # Common neighbors of u and v
            common_mask = edge_mask[u] & edge_mask[v]

            # Greedily extend
            while True:
                best_candidate = None
                best_score = -1

                temp = common_mask
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    temp ^= lsb

                    # Score by degree in remaining graph
                    score = bin(edge_mask[w] & common_mask).count('1')
                    if score > best_score:
                        best_score = score
                        best_candidate = w

                if best_candidate is None:
                    break

                current_clique.add(best_candidate)
                common_mask &= edge_mask[best_candidate]

            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)

            # Remove edges
            clique_list_sorted = sorted(current_clique)
            for i in range(len(clique_list_sorted)):
                u2 = clique_list_sorted[i]
                for j in range(i + 1, len(clique_list_sorted)):
                    v2 = clique_list_sorted[j]
                    edge_mask[u2] &= ~(1 << v2)
                    edge_mask[v2] &= ~(1 << u2)

            # Remove covered edges from remaining_edges
            clique_set = set(current_clique)
            new_remaining = []
            for e in remaining_edges:
                if e[0] in clique_set and e[1] in clique_set:
                    continue
                new_remaining.append(e)
            remaining_edges = new_remaining

        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)

    return results

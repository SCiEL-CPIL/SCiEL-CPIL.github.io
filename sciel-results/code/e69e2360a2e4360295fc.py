import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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

        # Pre-compute neighborhood masks for I vertices
        I_neigh_masks = {}
        for v in I:
            mask = 0
            for u in adj[v] & set(C):
                mask |= (1 << u)
            I_neigh_masks[v] = mask

        for v in I_sorted:
            N_v_mask = I_neigh_masks[v]
            N_v = set()
            temp = N_v_mask
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                N_v.add(u)
                temp ^= lsb

            if not N_v:
                continue

            while N_v:
                # Find a maximal clique in the induced subgraph on N_v
                # Use a greedy algorithm with bitmask operations
                
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
                # Update N_v_mask
                N_v_mask = 0
                for u in N_v:
                    N_v_mask |= (1 << u)
                I_neigh_masks[v] = N_v_mask

        # Remaining edges in C: partition into cliques
        # Use a greedy algorithm with edge-priority and local improvement

        # First, find all remaining edges
        remaining_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if edge_mask[u] & (1 << v):
                    remaining_edges.append((u, v))

        # If no remaining edges, we're done
        if not remaining_edges:
            return partition

        # Use a greedy algorithm: repeatedly find a maximal clique in the remaining graph
        # and add it to the partition. Use edge with max common neighborhood as seed.

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

            # Greedily extend: pick vertex with highest degree in common neighborhood
            while True:
                best_candidate = None
                best_score = -1

                temp = common_mask
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    temp ^= lsb

                    # Score by degree in remaining common neighborhood
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

        # Local improvement: try to merge pairs of cliques that form a larger clique
        # by checking if their union is a clique
        merged = True
        while merged:
            merged = False
            new_partition = []
            n_cliques = len(partition)
            for i in range(n_cliques):
                if i >= len(new_partition):
                    continue
                if i < len(new_partition) and new_partition[i] is None:
                    continue
                if i >= len(partition):
                    continue
                if i < len(new_partition) and new_partition[i] is None:
                    continue
                # Check if partition[i] was already merged
                if i < len(new_partition) and new_partition[i] is None:
                    continue
                cli = partition[i]
                if i < len(new_partition) and new_partition[i] is None:
                    continue
                # Try to merge with a later clique
                best_j = -1
                best_union_size = len(cli)
                for j in range(i + 1, n_cliques):
                    if j < len(new_partition) and new_partition[j] is None:
                        continue
                    clj = partition[j]
                    union = set(cli) | set(clj)
                    # Check if union is a clique: every pair must be adjacent
                    is_clique = True
                    for a in range(len(union)):
                        for b in range(a + 1, len(union)):
                            ua = sorted(union)[a]
                            ub = sorted(union)[b]
                            if ua >= k or ub >= k:
                                # One is in independent set, check adjacency
                                if ua < k and ub < k:
                                    if not (edge_mask[ua] & (1 << ub)) and (ua, ub) not in [(x, y) for x, y in edges if x < y]:
                                        # Check original adjacency
                                        pass
                                # For vertices involving I, they were only in cliques with their C-neighbors
                                # Since both cliques contain v (the I vertex), and union includes only C vertices + v
                                # Actually, cliques with I vertices contain v and some C vertices
                                # Two such cliques share v, so union = {v} ∪ (C1 ∪ C2)
                                # This is a clique iff C1 ∪ C2 is a clique in C
                                pass
                            else:
                                # Both in C
                                if not (edge_mask[ua] & (1 << ub)):
                                    # Edge was removed, meaning it's covered by some clique
                                    # But we need the union to be a clique in the ORIGINAL graph
                                    # Check original adjacency
                                    orig_adj = False
                                    for e in edges:
                                        if (e[0] == ua and e[1] == ub) or (e[0] == ub and e[1] == ua):
                                            orig_adj = True
                                            break
                                    if not orig_adj:
                                        is_clique = False
                                        break
                        if not is_clique:
                            break
                    if is_clique and len(union) > best_union_size:
                        best_union_size = len(union)
                        best_j = j
                if best_j >= 0:
                    union = sorted(set(partition[i]) | set(partition[best_j]))
                    new_partition.append(union)
                    new_partition.append(None)  # Mark best_j as merged
                    # Mark i as done (it's in new_partition)
                    merged = True
                else:
                    new_partition.append(cli)
            # Filter out None entries
            partition = [c for c in new_partition if c is not None]

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

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

        # Group independent vertices by identical neighborhood in C
        # This allows us to process them together, potentially finding larger cliques
        # that cover edges from multiple independent vertices at once
        groups = {}
        for v in I_sorted:
            N_v = adj[v] & set(C)
            if not N_v:
                continue
            key = frozenset(N_v)
            if key not in groups:
                groups[key] = []
            groups[key].append(v)

        # Process each group
        for N_key, v_list in groups.items():
            N_v = set(N_key)
            
            # For each vertex in the group, we need to cover its edges to C
            # But since they all have the same neighborhood, we can process them together
            # The key insight: if we find a clique S in N_v, we can create cliques
            # {v1} ∪ S, {v2} ∪ S, etc. for each v in v_list
            # This removes the same edges from C for all vertices in the group
            
            # Find maximal cliques in N_v to cover the edges
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

                # Add cliques for each vertex in the group
                for v in v_list:
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

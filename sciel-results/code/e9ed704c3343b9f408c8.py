import time
import random
import itertools

def construct():
    start_time = time.time()
    time_limit = 145.0

    def popcount(x):
        return bin(x).count('1')

    def is_regular_mask(mask, adj, n):
        k = popcount(mask)
        if k < 6:
            return True
        first_deg = -1
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            deg = popcount(adj[u] & mask)
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
            m ^= lsb
        return True

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def verify_full(adj, n):
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular_mask(mask, adj, n):
                    return False
        return True

    def solve_n(n, time_budget):
        end_time = start_time + time_budget
        if time.time() > end_time:
            return None

        subsets6 = []
        for comb in itertools.combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets6.append(mask)

        num_subsets = len(subsets6)

        edge_map = [[[] for _ in range(n)] for __ in range(n)]
        for i, mask in enumerate(subsets6):
            verts = []
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                verts.append(u)
                m ^= lsb
            for a in range(len(verts)):
                for b in range(a + 1, len(verts)):
                    u = verts[a]
                    v = verts[b]
                    edge_map[u][v].append(i)
                    edge_map[v][u].append(i)

        best_adj = None
        best_bad = float('inf')

        while time.time() < end_time:
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            bad = [0] * num_subsets
            total_bad = 0
            for i, mask in enumerate(subsets6):
                if is_regular_mask(mask, adj, n):
                    bad[i] = 1
                    total_bad += 1

            if total_bad < best_bad:
                best_bad = total_bad
                best_adj = adj[:]

            if total_bad == 0:
                if verify_full(adj, n):
                    return adj

            no_improve = 0
            for step in range(3000):
                if time.time() > end_time:
                    break
                if total_bad == 0:
                    if verify_full(adj, n):
                        return adj
                    break

                bad_indices = [i for i in range(num_subsets) if bad[i] == 1]
                if not bad_indices:
                    break

                # Score edges by how many bad subsets they're in
                edge_score = {}
                for idx in bad_indices:
                    mask = subsets6[idx]
                    verts = []
                    m = mask
                    while m:
                        lsb = m & -m
                        u = lsb.bit_length() - 1
                        verts.append(u)
                        m ^= lsb
                    for a in range(len(verts)):
                        for b in range(a + 1, len(verts)):
                            u, v = verts[a], verts[b]
                            if u > v:
                                u, v = v, u
                            key = (u, v)
                            edge_score[key] = edge_score.get(key, 0) + 1

                if edge_score:
                    best_edge = max(edge_score, key=edge_score.get)
                    u, v = best_edge
                else:
                    break

                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

                affected = edge_map[u][v]
                for i in affected:
                    old_bad = bad[i]
                    new_bad = 1 if is_regular_mask(subsets6[i], adj, n) else 0
                    if old_bad != new_bad:
                        bad[i] = new_bad
                        if old_bad:
                            total_bad -= 1
                        else:
                            total_bad += 1

                if total_bad < best_bad:
                    best_bad = total_bad
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve > 500:
                    break

        if best_adj is not None and verify_full(best_adj, n):
            return best_adj
        return None

    def try_extend(adj_base, n_base):
        """Try to add one vertex to a valid graph using constraint propagation."""
        end_time = start_time + 100.0
        n = n_base + 1
        new_v = n_base

        # All 5-subsets of original vertices
        five_subsets = list(itertools.combinations(range(n_base), 5))

        # For each 5-subset {v1..v5}, compute base degrees
        # and determine forced edge to new vertex
        vertex_forced = {}  # vertex -> set of allowed values {0,1}
        for v in range(n_base):
            vertex_forced[v] = {0, 1}

        for five in five_subsets:
            base_degs = []
            for i, vi in enumerate(five):
                deg = 0
                for j, vj in enumerate(five):
                    if i != j and (adj_base[vi] >> vj) & 1:
                        deg += 1
                base_degs.append(deg)

            sum_base = sum(base_degs)
            if sum_base % 4 != 0:
                # This subset cannot be regular regardless of edges to new vertex
                # Wait - that's wrong. The subset {v1..v5, new_v} has 6 vertices.
                # We need to check if it CAN be regular.
                # deg(new_v) = S = sum(e_i)
                # deg(vi) = base_deg_i + e_i
                # For regular: S = base_deg_i + e_i for all i
                # e_i = S - base_deg_i, and S = sum(e_i) = 5S - sum(base_degs)
                # 4S = sum(base_degs), so S = sum(base_degs)/4
                # e_i = S - base_deg_i must be in {0,1}
                # If sum_base % 4 != 0, no integer S exists, so this subset
                # will NEVER be regular. That's fine - no constraint.
                continue

            S = sum_base // 4
            for i, vi in enumerate(five):
                e_i = S - base_degs[i]
                if e_i < 0 or e_i > 1:
                    # This subset can never be regular - no constraint
                    break
                else:
                    vertex_forced[vi].discard(1 - e_i)
                    if not vertex_forced[vi]:
                        return None  # Infeasible

        # Check feasibility
        assignment = {}
        for v in range(n_base):
            if len(vertex_forced[v]) == 0:
                return None
            elif len(vertex_forced[v]) == 1:
                assignment[v] = list(vertex_forced[v])[0]
            else:
                assignment[v] = None

        undetermined = [v for v in range(n_base) if assignment[v] is None]

        if len(undetermined) > 12:
            # Random sampling
            for attempt in range(10000):
                if time.time() > end_time:
                    return None
                for v in undetermined:
                    assignment[v] = random.choice(list(vertex_forced[v]))

                adj = adj_base[:] + [0]
                for v in range(n_base):
                    if assignment[v]:
                        adj[v] |= (1 << new_v)
                        adj[new_v] |= (1 << v)

                # Verify all 6-subsets containing new_v
                ok = True
                for five in five_subsets:
                    mask = 0
                    for v in five:
                        mask |= (1 << v)
                    mask |= (1 << new_v)
                    if is_regular_mask(mask, adj, n):
                        ok = False
                        break
                if not ok:
                    continue

                # Full verification
                if verify_full(adj, n):
                    return adj
            return None
        else:
            # Exhaustive
            for bits in range(1 << len(undetermined)):
                if time.time() > end_time:
                    return None
                for idx, v in enumerate(undetermined):
                    assignment[v] = (bits >> idx) & 1

                adj = adj_base[:] + [0]
                for v in range(n_base):
                    if assignment[v]:
                        adj[v] |= (1 << new_v)
                        adj[new_v] |= (1 << v)

                ok = True
                for five in five_subsets:
                    mask = 0
                    for v in five:
                        mask |= (1 << v)
                    mask |= (1 << new_v)
                    if is_regular_mask(mask, adj, n):
                        ok = False
                        break
                if not ok:
                    continue

                if verify_full(adj, n):
                    return adj
            return None

    # Try to build 20-vertex graph
    adj20 = solve_n(20, 50.0)

    if adj20 is not None:
        # Try to extend to 21
        adj21 = try_extend(adj20, 20)
        if adj21 is not None:
            return {'n': 21, 'edges': get_edges_from_adj(adj21, 21)}
        return {'n': 20, 'edges': get_edges_from_adj(adj20, 20)}

    # Fallback: try 20 again with remaining time
    remaining = time_limit - (time.time() - start_time)
    if remaining > 10:
        adj20 = solve_n(20, remaining - 5)
        if adj20 is not None:
            return {'n': 20, 'edges': get_edges_from_adj(adj20, 20)}

    # Final fallback
    adj = [0] * 5
    for i in range(5):
        for j in range(i + 1, 5):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 5, 'edges': get_edges_from_adj(adj, 5)}

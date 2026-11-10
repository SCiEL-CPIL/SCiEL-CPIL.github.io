import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    def max_clique_in_induced(n, adj, mask, rng, max_size=None):
        best_clique = []
        found_large = [False]

        def expand(R, P, X):
            nonlocal best_clique
            if max_size and len(R) >= max_size:
                found_large[0] = True
                return
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return

            pc = popcount(P)
            if len(R) + pc <= len(best_clique):
                return

            PX = P | X
            max_deg = -1
            pivot = -1
            PX_copy = PX
            while PX_copy:
                v_bit = PX_copy & (-PX_copy)
                v = v_bit.bit_length() - 1
                PX_copy ^= v_bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    pivot = v

            if pivot == -1:
                return

            Q = P & ~adj[pivot]

            while Q:
                v_bit = Q & (-Q)
                v = v_bit.bit_length() - 1
                Q ^= v_bit

                new_R = R + [v]
                new_P = P & adj[v]
                new_X = X & adj[v]
                expand(new_R, new_P, new_X)
                if found_large[0]:
                    return

                P &= ~(1 << v)
                X |= (1 << v)
                Q = Q & P

        expand([], mask, 0)
        return best_clique

    def greedy_partition_lagrangian(n, adj, edge_list, deadline):
        """
        Lagrangian-relaxation guided greedy clique partition.
        """
        rem_adj = adj[:]
        active = bytearray([1]) * len(edge_list)
        cliques = []
        edge_weights = [1.0] * len(edge_list)
        vertex_pressure = [0.0] * n

        # Build edge index lookup: (u,v) -> index
        edge_index = {}
        for idx, (u, v) in enumerate(edge_list):
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx

        while True:
            if time.time() > deadline:
                return None

            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            best_idx = -1
            best_score = -1.0

            for idx in range(len(edge_list)):
                if not active[idx]:
                    continue
                u, v = edge_list[idx]
                if not (rem_adj[u] & (1 << v)):
                    active[idx] = 0
                    continue

                deg_u = popcount(rem_adj[u])
                deg_v = popcount(rem_adj[v])
                common = rem_adj[u] & rem_adj[v]
                cnt = popcount(common)

                if cnt == 0:
                    score = 1000.0
                else:
                    density = cnt / (deg_u + deg_v - cnt + 1)
                    pressure_pen = (vertex_pressure[u] + vertex_pressure[v]) * 0.3
                    size_bonus = cnt * 0.15
                    score = edge_weights[idx] * (density + size_bonus - pressure_pen)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == -1:
                break

            u, v = edge_list[best_idx]
            active[best_idx] = 0

            common_mask = rem_adj[u] & rem_adj[v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, None)
            clique = clique_core + [u, v]

            # Remove all edges within clique
            clique_set = set(clique)
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    rem_adj[a] &= ~(1 << b)
                    rem_adj[b] &= ~(1 << a)
                    key = (a, b) if a < b else (b, a)
                    if key in edge_index:
                        active[edge_index[key]] = 0

            # Update pressures
            for vertex in clique:
                vertex_pressure[vertex] += 0.05 * len(clique)

            # Decay weights for edges incident to clique vertices
            for vertex in clique:
                for idx in range(len(edge_list)):
                    if active[idx]:
                        eu, ev = edge_list[idx]
                        if eu == vertex or ev == vertex:
                            edge_weights[idx] *= 0.97

            cliques.append(clique)

        return cliques

    def local_improve(cliques, n, adj, deadline):
        if time.time() > deadline:
            return None

        current_cliques = [c[:] for c in cliques]
        best_count = len(current_cliques)
        improved = True

        while improved and time.time() < deadline:
            improved = False
            best_new = None
            best_new_count = best_count

            for i in range(len(current_cliques)):
                if time.time() > deadline:
                    return None
                ci = current_cliques[i]
                for j in range(i + 1, len(current_cliques)):
                    if time.time() > deadline:
                        return None
                    cj = current_cliques[j]
                    merged = ci + cj
                    is_clique = True
                    for a in range(len(merged)):
                        for b in range(a + 1, len(merged)):
                            u, v = merged[a], merged[b]
                            if not (adj[u] & (1 << v)):
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique and len(merged) >= 2:
                        new_cliques = [c for k, c in enumerate(current_cliques) if k != i and k != j]
                        new_cliques.append(merged)
                        if len(new_cliques) < best_new_count:
                            best_new_count = len(new_cliques)
                            best_new = new_cliques
                        improved = True

            if best_new is not None:
                current_cliques = best_new
                best_count = best_new_count
            else:
                improved = False

        return current_cliques

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        edge_list = edges[:]

        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)

        start = time.time()
        attempt = 0
        while time.time() < start + time_limit:
            rng = random.Random(seed * 1000 + attempt)
            result = greedy_partition_lagrangian(n, adj, edge_list, DEADLINE)
            if result is not None:
                improved = local_improve(result, n, adj, DEADLINE)
                if improved is not None:
                    count = len(improved)
                    if count < best_count:
                        best_count = count
                        best_cliques = improved
                else:
                    count = len(result)
                    if count < best_count:
                        best_count = count
                        best_cliques = result
            attempt += 1

        return best_cliques

    t1 = 15.0
    t2 = 90.0

    current = time.time()
    remaining = DEADLINE - current
    if t1 + t2 > remaining:
        scale = remaining / (t1 + t2)
        t1 *= scale
        t2 *= scale

    result1 = solve_instance(32, 0.6, 601, t1)

    if time.time() > DEADLINE:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, t2)

    return [result1, result2]

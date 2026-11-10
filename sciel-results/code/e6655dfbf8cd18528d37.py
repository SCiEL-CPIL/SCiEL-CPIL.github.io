import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0

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

    def triangle_2matching_greedy(n, adj, rng, deadline):
        """
        Greedy maximum 2-matching: repeatedly find triangles (3-cliques)
        that are edge-disjoint, preferring those with high 'value'.
        A 2-matching is a set of edges where each vertex has degree <= 2.
        In a triangle, each vertex has degree 2, so triangles are 2-matchings.
        We greedily pick triangles to maximize coverage of edges in 3-cliques.
        """
        rem_adj = adj[:]
        triangles = []

        while True:
            if time.time() > deadline:
                break

            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            # Find all triangles in remaining graph and pick the best one
            # A triangle (u,v,w) has u<v<w, and all three edges present
            best_tri = None
            best_score = -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                # Get neighbors of u with index > u
                neigh_u = rem_adj[u] & (~((1 << (u + 1)) - 1))
                if not neigh_u:
                    continue
                # For each pair of neighbors v, w of u (v < w), check if edge (v,w) exists
                v_list = []
                tmp = neigh_u
                while tmp:
                    bit = tmp & (-tmp)
                    v_list.append(bit.bit_length() - 1)
                    tmp ^= bit

                for i in range(len(v_list)):
                    v = v_list[i]
                    if not (rem_adj[v] & (1 << u)):
                        continue
                    # Neighbors of v with index > v
                    neigh_v = rem_adj[v] & (~((1 << (v + 1)) - 1))
                    # Need to also have u as neighbor, but we already checked
                    # Check common neighbors of u and v that are > v
                    common = neigh_u & neigh_v
                    if not common:
                        continue
                    # Iterate common neighbors
                    tmp2 = common
                    while tmp2:
                        bit = tmp2 & (-tmp2)
                        w = bit.bit_length() - 1
                        tmp2 ^= bit
                        # Triangle (u, v, w)
                        # Score: prefer triangles where vertices have high remaining degree
                        # (so we remove more "valuable" edges)
                        du = popcount(rem_adj[u])
                        dv = popcount(rem_adj[v])
                        dw = popcount(rem_adj[w])
                        score = du + dv + dw + rng.random() * 0.1
                        if score > best_score:
                            best_score = score
                            best_tri = (u, v, w)

            if best_tri is None:
                break

            u, v, w = best_tri
            triangles.append([u, v, w])

            # Remove the three edges of this triangle
            rem_adj[u] &= ~(1 << v)
            rem_adj[v] &= ~(1 << u)
            rem_adj[u] &= ~(1 << w)
            rem_adj[w] &= ~(1 << u)
            rem_adj[v] &= ~(1 << w)
            rem_adj[w] &= ~(1 << v)

        return triangles

    def complete_partition(n, adj, triangles, deadline):
        """
        Given a set of triangles (3-cliques), complete the partition by
        covering remaining edges with 2-cliques (single edges).
        Returns full list of cliques.
        """
        rem_adj = adj[:]
        cliques = []

        # Remove triangle edges
        for tri in triangles:
            u, v, w = tri[0], tri[1], tri[2]
            rem_adj[u] &= ~(1 << v)
            rem_adj[v] &= ~(1 << u)
            rem_adj[u] &= ~(1 << w)
            rem_adj[w] &= ~(1 << u)
            rem_adj[v] &= ~(1 << w)
            rem_adj[w] &= ~(1 << v)
            cliques.append(tri)

        # Cover remaining edges as 2-cliques
        for u in range(n):
            if not rem_adj[u]:
                continue
            v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
            while v_mask:
                bit = v_mask & (-v_mask)
                v = bit.bit_length() - 1
                v_mask ^= bit
                cliques.append([u, v])
                rem_adj[u] &= ~(1 << v)
                rem_adj[v] &= ~(1 << u)

        return cliques

    def max_clique_in_induced(n, adj, mask, rng):
        """
        Find max clique in G[mask].
        """
        best_clique = []

        def expand(R, P):
            nonlocal best_clique
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return

            pc = popcount(P)
            if len(R) + pc <= len(best_clique):
                return

            # Pivot selection
            max_deg = -1
            candidates = []
            P_copy = P
            while P_copy:
                v_bit = P_copy & (-P_copy)
                v = v_bit.bit_length() - 1
                P_copy ^= v_bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [v]
                elif deg == max_deg:
                    candidates.append(v)

            if not candidates:
                return

            u = rng.choice(candidates)
            Q = P & ~adj[u]

            while Q:
                max_v_deg = -1
                v_candidates = []
                Q_copy = Q
                while Q_copy:
                    v_bit = Q_copy & (-Q_copy)
                    i = v_bit.bit_length() - 1
                    Q_copy ^= v_bit
                    deg = popcount(P & adj[i])
                    if deg > max_v_deg:
                        max_v_deg = deg
                        v_candidates = [i]
                    elif deg == max_v_deg:
                        v_candidates.append(i)

                if not v_candidates:
                    break

                v = rng.choice(v_candidates)

                new_R = R + [v]
                new_P = P & adj[v]
                expand(new_R, new_P)

                P &= ~(1 << v)
                Q = Q & P

        expand([], mask)
        return best_clique

    def greedy_partition_max(n, adj, rng, deadline):
        """
        Greedy: repeatedly find edge with MAX common neighbors,
        extract max clique containing it, remove those edges.
        """
        rem_adj = adj[:]
        cliques = []

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

            max_common = -1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt > max_common:
                        max_common = cnt
                        best_u, best_v = u, v

            if best_u == -1:
                break

            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def hybrid_partition(n, adj, rng, deadline):
        """
        Hybrid: First extract a set of triangles via 2-matching greedy,
        then complete with max-clique greedy on remaining edges.
        """
        rem_adj = adj[:]
        cliques = []

        # Phase 1: Extract triangles
        triangles = []
        while True:
            if time.time() > deadline:
                break

            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            best_tri = None
            best_score = -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                neigh_u = rem_adj[u] & (~((1 << (u + 1)) - 1))
                if not neigh_u:
                    continue
                v_list = []
                tmp = neigh_u
                while tmp:
                    bit = tmp & (-tmp)
                    v_list.append(bit.bit_length() - 1)
                    tmp ^= bit

                for i in range(len(v_list)):
                    v = v_list[i]
                    if not (rem_adj[v] & (1 << u)):
                        continue
                    neigh_v = rem_adj[v] & (~((1 << (v + 1)) - 1))
                    common = neigh_u & neigh_v
                    if not common:
                        continue
                    tmp2 = common
                    while tmp2:
                        bit = tmp2 & (-tmp2)
                        w = bit.bit_length() - 1
                        tmp2 ^= bit
                        du = popcount(rem_adj[u])
                        dv = popcount(rem_adj[v])
                        dw = popcount(rem_adj[w])
                        score = du + dv + dw + rng.random() * 0.1
                        if score > best_score:
                            best_score = score
                            best_tri = (u, v, w)

            if best_tri is None:
                break

            u, v, w = best_tri
            cliques.append([u, v, w])
            rem_adj[u] &= ~(1 << v)
            rem_adj[v] &= ~(1 << u)
            rem_adj[u] &= ~(1 << w)
            rem_adj[w] &= ~(1 << u)
            rem_adj[v] &= ~(1 << w)
            rem_adj[w] &= ~(1 << v)

        # Phase 2: Complete with max-clique greedy
        while True:
            if time.time() > deadline:
                break

            has_edges = False
            for i in range(n):
                if rem_adj[i]:
                    has_edges = True
                    break
            if not has_edges:
                break

            max_common = -1
            best_u, best_v = -1, -1

            for u in range(n):
                if not rem_adj[u]:
                    continue
                v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    v_bit = v_mask & (-v_mask)
                    v = v_bit.bit_length() - 1
                    v_mask ^= v_bit

                    common = rem_adj[u] & rem_adj[v]
                    cnt = popcount(common)
                    if cnt > max_common:
                        max_common = cnt
                        best_u, best_v = u, v

            if best_u == -1:
                break

            common_mask = rem_adj[best_u] & rem_adj[best_v]
            clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
            clique = clique_core + [best_u, best_v]

            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)

            cliques.append(clique)

        return cliques

    def validate_partition(n, edges, cliques):
        """Validate that cliques form a valid partition."""
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))

        covered = set()
        for clique in cliques:
            if len(clique) < 2:
                return False
            # Check clique is valid
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (min(u, v), max(u, v))
                    if e not in edge_set:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)

        return covered == edge_set

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)

        # Baseline: each edge as its own 2-clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)

        start = time.time()
        attempt = 0
        while time.time() < start + time_limit:
            rng = random.Random(seed * 1000 + attempt)

            if attempt % 3 == 0:
                result = hybrid_partition(n, adj, rng, DEADLINE)
            elif attempt % 3 == 1:
                result = greedy_partition_max(n, adj, rng, DEADLINE)
            else:
                # Triangle-first then complete with 2-cliques only
                triangles = triangle_2matching_greedy(n, adj, rng, DEADLINE)
                result = complete_partition(n, adj, triangles, DEADLINE)

            if result is not None:
                # Validate
                if validate_partition(n, edges, result):
                    count = len(result)
                    if count < best_count:
                        best_count = count
                        best_cliques = result
            attempt += 1

        return best_cliques

    # Time budget allocation
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

import time
import random

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

    def max_clique_bronk(n, adj, mask, rng, max_size=None):
        """Bron-Kerbosch with pivot to find max clique in G[mask]."""
        best = []

        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = R[:]
                return
            if len(R) + popcount(P) <= len(best):
                return

            # Pivot: choose vertex with max degree in P
            pivot_candidates = P | X
            if not pivot_candidates:
                if len(R) > len(best):
                    best = R[:]
                return

            # Find pivot with max neighbors in P
            max_deg = -1
            pivot = -1
            pc = pivot_candidates
            while pc:
                bit = pc & (-pc)
                v = bit.bit_length() - 1
                pc ^= bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    pivot = v

            # Iterate over P \ N(pivot)
            candidates = P & ~adj[pivot] if pivot >= 0 else P
            if not candidates:
                candidates = P

            c = candidates
            while c:
                bit = c & (-c)
                v = bit.bit_length() - 1
                c ^= bit

                bk(R + [v], P & adj[v], X & adj[v])
                P &= ~(1 << v)
                X |= (1 << v)

                if time.time() > DEADLINE:
                    return

        bk([], mask, 0)
        return best

    def greedy_partition_bitmask(n, adj, rng, deadline, strategy="min_common"):
        """
        Greedy clique partition using bitmask adjacency.
        strategy: 'min_common' - pick edge with fewest common neighbors first
                  'max_degree' - pick edge incident to highest degree vertex
                  'random' - pick random edge
        """
        rem_adj = adj[:]
        cliques = []

        # Helper to check if edges remain
        def has_edges():
            for i in range(n):
                if rem_adj[i]:
                    return True
            return False

        # Helper to pick next edge
        def pick_edge():
            if strategy == "min_common":
                min_common = n + 1
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
                        if cnt < min_common:
                            min_common = cnt
                            best_u, best_v = u, v
                            if min_common == 0:
                                return u, v
                return best_u, best_v
            elif strategy == "max_degree":
                # Find vertex with max degree
                max_deg = -1
                best_v = -1
                for v in range(n):
                    d = popcount(rem_adj[v])
                    if d > max_deg:
                        max_deg = d
                        best_v = v
                if best_v < 0:
                    return -1, -1
                # Pick a random neighbor
                nbrs = rem_adj[best_v]
                if not nbrs:
                    return -1, -1
                bits = []
                m = nbrs
                while m:
                    b = m & (-m)
                    bits.append(b.bit_length() - 1)
                    m ^= b
                return best_v, rng.choice(bits)
            else:  # random
                # Collect all edges
                edge_list = []
                for u in range(n):
                    v_mask = rem_adj[u] & (~((1 << (u + 1)) - 1))
                    while v_mask:
                        v_bit = v_mask & (-v_mask)
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit
                        edge_list.append((u, v))
                if not edge_list:
                    return -1, -1
                return rng.choice(edge_list)

        while True:
            if time.time() > deadline:
                return None
            if not has_edges():
                break

            u, v = pick_edge()
            if u < 0 or v < 0:
                break

            # Find max clique containing u and v
            common_mask = rem_adj[u] & rem_adj[v]
            clique_core = max_clique_bronk(n, rem_adj, common_mask, rng)
            clique = clique_core + [u, v]

            if len(clique) < 2:
                clique = [u, v]

            # Remove edges of this clique from rem_adj
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    rem_adj[a] &= ~(1 << b)
                    rem_adj[b] &= ~(1 << a)

            cliques.append(clique)

        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)

        # Baseline: each edge as its own 2-clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)

        start = time.time()
        attempt = 0
        strategies = ["min_common", "max_degree", "random"]

        while time.time() < start + time_limit:
            if time.time() > DEADLINE:
                break

            strategy = strategies[attempt % len(strategies)]
            rng = random.Random(seed * 1000 + attempt * 7)
            result = greedy_partition_bitmask(n, adj, rng, DEADLINE, strategy)
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
            attempt += 1

        return best_cliques

    # Time budget allocation
    t1 = 20.0
    t2 = 85.0

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

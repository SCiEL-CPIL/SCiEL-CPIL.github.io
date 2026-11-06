import time
from random import Random

def construct():
    deadline = time.time() + 140  # Safety margin

    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    # Precompute popcount for 16-bit chunks
    pc16 = [0] * 65536
    for i in range(65536):
        pc16[i] = bin(i).count('1')
        
    def popcount(x):
        # x is up to 64 bits
        c = 0
        while x:
            c += pc16[x & 0xFFFF]
            x >>= 16
        return c

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []

        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        remaining = set(edges)
        cliques = []
        time_limit = deadline - 10

        def find_max_clique(adj, n, active_mask, time_limit):
            """Find a maximum clique in the subgraph defined by adj and active_mask."""
            best_clique = []
            best_size = 0

            def bk(R, P, X):
                nonlocal best_clique, best_size
                if time.time() > time_limit:
                    return
                if P == 0 and X == 0:
                    if len(R) > best_size:
                        best_size = len(R)
                        best_clique = list(R)
                    return
                if P == 0:
                    return

                # Choose pivot: vertex in P∪X with max neighbors in P
                pivot_set = P | X
                pivot = -1
                pivot_deg = -1
                tmp = pivot_set
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    d = popcount(adj[v] & P)
                    if d > pivot_deg:
                        pivot_deg = d
                        pivot = v
                    tmp ^= lsb

                # Branch on P \ N(pivot)
                candidates = P & ~(adj[pivot] if pivot >= 0 else 0)
                tmp = candidates
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    if time.time() > time_limit:
                        return
                    R.append(v)
                    bk(R, P & adj[v], X & adj[v])
                    R.pop()
                    P &= ~(1 << v)
                    X |= (1 << v)

            bk([], active_mask, 0)
            return best_clique

        while remaining:
            if time.time() > time_limit:
                # Emergency: pack remaining edges as pairs
                for e in list(remaining):
                    cliques.append(list(e))
                break

            # Build subgraph of remaining edges
            sub_adj = [0] * n
            active_mask = 0
            for u, v in remaining:
                sub_adj[u] |= (1 << v)
                sub_adj[v] |= (1 << u)
                active_mask |= (1 << u) | (1 << v)

            if active_mask == 0:
                break

            clique = find_max_clique(sub_adj, n, active_mask, time_limit)

            if not clique or len(clique) < 2:
                # Fallback: take any edge
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                continue

            # Remove all edges within the clique from remaining
            cset = set(clique)
            to_remove = []
            for e in remaining:
                if e[0] in cset and e[1] in cset:
                    to_remove.append(e)

            if to_remove:
                cliques.append(clique)
                for e in to_remove:
                    remaining.discard(e)
            else:
                # Fallback
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = random_graph(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result

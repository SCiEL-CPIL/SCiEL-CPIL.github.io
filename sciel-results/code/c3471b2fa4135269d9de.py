import time
from random import Random
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Bitmask adjacency of the remaining-edge graph
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        cliques = []

        # Tomita-style maximum clique via branch and bound with pivot.
        # Works on the current (remaining-edge) graph defined by adj bitmasks.
        def max_clique(adj, n):
            best = []
            best_size = 0
            node_count = 0
            deadline_local = deadline - 15

            def popcount(x):
                return bin(x).count('1')

            def expand(R, P, X):
                nonlocal best, best_size, node_count
                if time.time() > deadline_local:
                    return
                node_count += 1
                if node_count % 4096 == 0 and time.time() > deadline_local:
                    return
                if P == 0:
                    if X == 0 and len(R) > best_size:
                        best_size = len(R)
                        best = list(R)
                    return
                # Upper bound: |R| + |P|
                if len(R) + popcount(P) <= best_size:
                    return
                # Pivot: choose v in P|X maximizing |N(v) & P|
                union = P | X
                if union == 0:
                    return
                best_pivot = -1
                best_pivot_deg = -1
                tmp = union
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    d = popcount(adj[v] & P)
                    if d > best_pivot_deg:
                        best_pivot_deg = d
                        best_pivot = v
                    tmp ^= lsb
                # Candidates: P minus neighbors of pivot and pivot itself
                candidates = P & ~(adj[best_pivot] | (1 << best_pivot))
                while candidates:
                    lsb = candidates & (-candidates)
                    v = lsb.bit_length() - 1
                    candidates ^= lsb
                    R.append(v)
                    expand(R, P & adj[v], X & adj[v])
                    R.pop()
                    P ^= lsb
                    X |= lsb
                    if time.time() > deadline_local:
                        return

            expand([], (1 << n) - 1, 0)
            return best if len(best) >= 2 else None

        # Greedy clique fallback (fast, no recursion)
        def greedy_clique(adj, n):
            active = [v for v in range(n) if adj[v] != 0]
            if not active:
                return None
            active.sort(key=lambda v: bin(adj[v]).count('1'), reverse=True)
            best = None
            best_size = 0
            for start in active[:30]:
                clique = [start]
                P = adj[start] & ~(1 << start)
                while P:
                    best_v = -1
                    best_d = -1
                    tmp = P
                    while tmp:
                        lsb = tmp & (-tmp)
                        v = lsb.bit_length() - 1
                        d = bin(adj[v] & P).count('1')
                        if d > best_d:
                            best_d = d
                            best_v = v
                        tmp ^= lsb
                    if best_v < 0:
                        break
                    clique.append(best_v)
                    P = (P & adj[best_v]) & ~(1 << best_v)
                if len(clique) > best_size:
                    best_size = len(clique)
                    best = clique
            return best if best and len(best) >= 2 else None

        # Track remaining edges as a set of (min,max) tuples
        remaining = set((min(u, v), max(u, v)) for u, v in edges)

        while remaining:
            if time.time() > deadline - 20:
                for e in list(remaining):
                    cliques.append(list(e))
                break

            # Find a large clique in the remaining-edge graph
            clique = max_clique(adj, n)

            # Fallback to greedy if max_clique timed out or found nothing
            if clique is None or len(clique) < 2:
                clique = greedy_clique(adj, n)

            if clique is None or len(clique) < 2:
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
                continue

            # Remove all edges inside the clique from remaining and adj
            cset = set(clique)
            to_remove = set()
            for u, v in remaining:
                if u in cset and v in cset:
                    to_remove.add((u, v))
            if to_remove:
                cliques.append(clique)
                for u, v in to_remove:
                    remaining.discard((u, v))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result

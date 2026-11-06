import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def maximal_cliques(n, adj, edge_set):
        # Bron-Kerbosch with pivot; returns list of frozensets (maximal cliques of size>=2)
        cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # choose pivot u from P union X maximizing |P ∩ N(u)|
            Pux = P | X
            best_u = None
            best_cnt = -1
            for u in Pux:
                c = len(P & adj[u])
                if c > best_cnt:
                    best_cnt = c
                    best_u = u
            # iterate over P \ N(u)
            if best_u is None:
                candidates = list(P)
            else:
                candidates = [v for v in P if v not in adj[best_u]]
            for v in candidates:
                newR = R | {v}
                newP = P & adj[v]
                newX = X & adj[v]
                bk(newR, newP, newX)
                P = P - {v}
                X = X | {v}

        bk(set(), set(range(n)), set())
        return cliques

    def solve(edges):
        if not edges:
            return []

        n = max(max(e) for e in edges) + 1
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))

        # Precompute edge list for each maximal clique
        mc = maximal_cliques(n, adj, edge_set)
        # For each maximal clique, its internal edges
        clique_edges = []
        for c in mc:
            verts = list(c)
            el = []
            for u, v in combinations(verts, 2):
                e = (u, v) if u < v else (v, u)
                if e in edge_set:
                    el.append(e)
            if el:
                clique_edges.append((c, el))

        # Greedy set cover: repeatedly pick clique covering most uncovered edges
        remaining = set(edge_set)
        chosen = []

        # Precompute for speed: map edge -> list of clique indices
        edge_to_cliques = {e: [] for e in edge_set}
        for idx, (c, el) in enumerate(clique_edges):
            for e in el:
                edge_to_cliques[e].append(idx)

        # For incremental scoring, keep count of how many of clique's edges are still uncovered
        # We'll just recompute scores each round but only over candidate cliques that have uncovered edges.
        # To make it fast, maintain per-clique uncovered count.
        uncov_count = [len(el) for (c, el) in clique_edges]
        # Also maintain, for each clique, the set of its edges (for removal)
        # We track a "alive" flag.

        alive = [True] * len(clique_edges)

        # Edge -> alive cliques that cover it (maintain as list; we filter lazily)
        # Simpler: each round, build list of (count, idx) for alive cliques with count>0, pick max.
        # n_cliques is small; m edges up to ~1000. Rounds <= m. This is fine.

        while remaining:
            if time.time() > deadline:
                # Fallback: cover remaining edges one by one as 2-cliques
                for e in remaining:
                    chosen.append(list(e))
                break

            # Find best clique
            best_idx = -1
            best_cnt = 0
            for idx in range(len(clique_edges)):
                if not alive[idx]:
                    continue
                cnt = uncov_count[idx]
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = idx

            if best_idx == -1 or best_cnt == 0:
                # No maximal clique covers any remaining edge (shouldn't happen since every edge
                # is in some maximal clique). Fallback to single edge.
                e = next(iter(remaining))
                chosen.append(list(e))
                remaining.discard(e)
                # decrement counts for cliques containing e
                for idx in edge_to_cliques[e]:
                    if alive[idx]:
                        uncov_count[idx] -= 1
                        if uncov_count[idx] == 0:
                            alive[idx] = False
                continue

            c, el = clique_edges[best_idx]
            chosen.append(list(c))
            alive[best_idx] = False
            # Remove these edges from remaining and update counts
            for e in el:
                if e in remaining:
                    remaining.discard(e)
            for e in el:
                for idx in edge_to_cliques[e]:
                    if alive[idx] and idx != best_idx:
                        uncov_count[idx] -= 1
                        if uncov_count[idx] == 0:
                            alive[idx] = False
            uncov_count[best_idx] = 0

        return chosen

    c1 = solve(random_graph(32, 0.6, 601))
    c2 = solve(random_graph(64, 0.55, 602))

    return [c1, c2]

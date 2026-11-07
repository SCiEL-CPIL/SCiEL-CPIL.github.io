import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def bron_kerbosch_max_clique(adj, vertices):
        """
        Find a maximum clique in the graph defined by adj (list of sets) on vertices.
        Uses Bron-Kerbosch with pivot for efficiency.
        Returns the largest clique found as a list.
        """
        if not vertices:
            return []

        best = []
        vertices_set = set(vertices)

        def bk(R, P, X):
            nonlocal best
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return

            # Pivot selection: choose vertex from P∪X with max degree in P
            pivot_candidates = list(P | X)
            if pivot_candidates:
                # Pick pivot that maximizes |P ∩ N(u)|
                pivot = max(pivot_candidates, key=lambda u: len(P & adj[u]))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = list(P)

            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                P.remove(v)
                X.add(v)
                bk(new_R, new_P, new_X)
                P.add(v)
                X.remove(v)

        bk(set(), vertices_set, set())
        return best

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj_k = [set() for _ in range(k)]
        neighbors_in_k = {v: set() for v in range(k, n)}

        for u, v in edges:
            if u < k and v < k:
                adj_k[u].add(v)
                adj_k[v].add(u)
            elif u >= k and v < k:
                neighbors_in_k[u].add(v)
            elif v >= k and u < k:
                neighbors_in_k[v].add(u)

        used_edges = set()
        cliques = []

        def norm_edge(u, v):
            return (min(u, v), max(u, v))

        # Process each v in I: find max clique in available K-neighbors
        # Sort I vertices by decreasing degree to cover more edges first
        i_vertices = sorted(range(k, n), key=lambda v: -len(neighbors_in_k[v]))

        for v in i_vertices:
            if time.time() > deadline:
                # Fallback: cover remaining cross edges with pairs
                for u in neighbors_in_k[v]:
                    if norm_edge(u, v) not in used_edges:
                        cliques.append([v, u])
                        used_edges.add(norm_edge(u, v))
                continue

            N_v = set(neighbors_in_k[v])
            # Filter to only those whose edge to v is not yet used
            # (all cross edges from v should be unused initially since we process each v once)
            remaining = list(N_v)

            while remaining:
                if time.time() > deadline:
                    for u in remaining:
                        e = norm_edge(v, u)
                        if e not in used_edges:
                            cliques.append([v, u])
                            used_edges.add(e)
                    break

                m = len(remaining)
                if m == 1:
                    u = remaining[0]
                    e = norm_edge(v, u)
                    if e not in used_edges:
                        cliques.append([v, u])
                        used_edges.add(e)
                    remaining = []
                    continue

                # Build adjacency for the "available" graph on remaining
                # Two K-vertices u, w are connected if edge (u,w) is NOT in used_edges
                rem_set = set(remaining)
                adj_avail = [set() for _ in range(m)]
                idx_map = {v_idx: i for i, v_idx in enumerate(remaining)}

                for i in range(m):
                    for j in range(i + 1, m):
                        u, w = remaining[i], remaining[j]
                        e = norm_edge(u, w)
                        if e not in used_edges:
                            adj_avail[i].add(j)
                            adj_avail[j].add(i)

                # Find maximum clique in adj_avail
                best_local = bron_kerbosch_max_clique(adj_avail, list(range(m)))

                if not best_local:
                    best_local = [0]

                S = [remaining[idx] for idx in best_local]

                # Add clique [v] + S
                cliques.append([v] + S)

                # Mark all edges in this clique as used
                # Cross edges: (v, u) for u in S
                for u in S:
                    used_edges.add(norm_edge(v, u))
                # K-edges: (u, w) for u, w in S
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        used_edges.add(norm_edge(S[i], S[j]))

                # Remove S from remaining
                S_set = set(S)
                remaining = [x for x in remaining if x not in S_set]

        # Cover remaining K-edges
        E_rem = set()
        for u in range(k):
            for w in range(u + 1, k):
                if norm_edge(u, w) not in used_edges:
                    E_rem.add(norm_edge(u, w))

        if E_rem:
            while E_rem:
                if time.time() > deadline:
                    # Fallback: add edges as pairs
                    for e in E_rem:
                        cliques.append([e[0], e[1]])
                    break

                # Find active vertices
                active_vertices = set()
                for u, w in E_rem:
                    active_vertices.add(u)
                    active_vertices.add(w)

                v_list = list(active_vertices)
                m = len(v_list)
                idx_map = {v: i for i, v in enumerate(v_list)}

                # Build adjacency for remaining K-edges
                adj_avail = [set() for _ in range(m)]
                for u, w in E_rem:
                    i = idx_map[u]
                    j = idx_map[w]
                    adj_avail[i].add(j)
                    adj_avail[j].add(i)

                best_local = bron_kerbosch_max_clique(adj_avail, list(range(m)))

                if not best_local:
                    # Take any single edge
                    u, w = next(iter(E_rem))
                    S = [u, w]
                else:
                    S = [v_list[idx] for idx in best_local]

                cliques.append(S)

                # Remove all edges within S from E_rem
                S_set = set(S)
                edges_to_remove = set()
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = norm_edge(S[i], S[j])
                        if e in E_rem:
                            edges_to_remove.add(e)
                E_rem -= edges_to_remove

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))

    for spec in specs:
        res = solve_one(*spec)
        results.append(res)

    return results

# EVOLVE-BLOCK-END

import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_max_clique(vertices, adj, time_limit=None):
        """Bron-Kerbosch with pivot to find a maximum clique."""
        best = []
        vertices_set = set(vertices)

        def bk(R, P, X):
            nonlocal best
            if time_limit is not None and time.time() > time_limit:
                return
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if not P:
                return
            # Pivot: maximize |N(u) ∩ P|
            pivot = None
            max_count = -1
            for u in (P | X):
                c = len(adj[u] & P)
                if c > max_count:
                    max_count = c
                    pivot = u
            if pivot is None:
                return
            candidates = P - adj[pivot]
            for v in list(candidates):
                if time_limit is not None and time.time() > time_limit:
                    return
                bk(R | {v}, P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)

        bk(set(), vertices_set, set())
        return best

    def greedy_clique(seed_vertex, remaining_edges, remaining_adj):
        """Greedy grow a clique from seed_vertex using only remaining edges."""
        clique = [seed_vertex]
        candidates = set(remaining_adj[seed_vertex])
        while candidates:
            # Pick candidate with max degree in remaining graph
            best = max(candidates, key=lambda x: len(remaining_adj[x]))
            # Verify it connects to all in clique via remaining edges
            ok = all((min(best, c), max(best, c)) in remaining_edges for c in clique)
            if ok:
                clique.append(best)
                candidates.discard(best)
                # Shrink candidates to common neighbors in remaining graph
                candidates = {x for x in candidates if all(
                    (min(x, c), max(x, c)) in remaining_edges for c in clique)}
            else:
                candidates.discard(best)
        return clique

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []

        adj = build_adj(edges, n)
        remaining_edges = set(edges)
        remaining_adj = [set() for _ in range(n)]
        for u, v in remaining_edges:
            remaining_adj[u].add(v)
            remaining_adj[v].add(u)

        partition = []
        deadline = start_time + time_budget

        while remaining_edges:
            if time.time() > deadline:
                # Fallback: dump remaining edges as K2s
                for u, v in remaining_edges:
                    partition.append([u, v])
                break

            # Find a maximum clique in the residual graph
            active = set()
            for u, v in remaining_edges:
                active.add(u)
                active.add(v)

            # Time-box the max-clique search; use greedy if it's too slow
            bk_deadline = min(time.time() + 2.0, deadline - 1.0)
            max_c = find_max_clique(active, remaining_adj, bk_deadline)

            if not max_c or len(max_c) < 2:
                # Fallback to greedy from a high-degree vertex
                sv = max(active, key=lambda x: len(remaining_adj[x]))
                max_c = greedy_clique(sv, remaining_edges, remaining_adj)
                if len(max_c) < 2:
                    u, v = next(iter(remaining_edges))
                    max_c = [u, v]

            partition.append(max_c)

            # Remove clique edges from remaining
            cs = set(max_c)
            for i in range(len(max_c)):
                for j in range(i + 1, len(max_c)):
                    e = (min(max_c[i], max_c[j]), max(max_c[i], max_c[j]))
                    remaining_edges.discard(e)
                    remaining_adj[max_c[i]].discard(max_c[j])
                    remaining_adj[max_c[j]].discard(max_c[i])

        # Refinement: merge disjoint cliques into larger ones where possible
        # (only helps if the residual allowed splitting; attempt a few passes)
        if time.time() < deadline - 3.0:
            for _ in range(3):
                merged = []
                used = [False] * len(partition)
                for i in range(len(partition)):
                    if used[i]:
                        continue
                    used[i] = True
                    cur = set(partition[i])
                    for j in range(i + 1, len(partition)):
                        if used[j]:
                            continue
                        # Can we add all of partition[j] to cur?
                        trial = cur | set(partition[j])
                        ok = True
                        for a in range(len(trial)):
                            for b in range(a + 1, len(trial)):
                                if (min(trial[a], trial[b]), max(trial[a], trial[b])) not in edges:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            cur = trial
                            used[j] = True
                    merged.append(list(cur))
                if len(merged) >= len(partition):
                    break
                partition = merged

        return partition

    # Solve both instances with time budgets
    t1 = time.time()
    budget1 = min(60, DEADLINE - (t1 - start_time) - 15)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(60, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]

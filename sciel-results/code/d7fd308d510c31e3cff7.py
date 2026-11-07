"""Rewritten program: clique cover of split graphs via structured set-cover."""
import time

# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _cover_one(n, k, p, seed, deadline):
        rng = Random(seed)
        S = [set() for _ in range(n)]
        # C internal
        for u in range(k):
            for v in range(u + 1, k):
                S[u].add(v)
                S[v].add(u)
        # cross
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    S[u].add(v)
                    S[v].add(u)
        edges = []
        for u in range(n):
            for v in S[u]:
                if u < v:
                    edges.append((u, v))

        C = list(range(k))
        I = list(range(k, n))
        I_neigh = {v: S[v] & set(C) for v in I}

        # precompute C-internal edge sets
        C_edge_sets = {}
        for a in range(k):
            for b in range(a + 1, k):
                C_edge_sets[(a, b)] = {a, b}

        # edge -> list of I vertices incident to it (only cross edges matter for grouping)
        edge_to_iv = {}
        for v in I:
            for u in I_neigh[v]:
                key = (u, v)
                edge_to_iv.setdefault(key, []).append(v)

        def clique_edges(clique):
            es = set()
            L = list(clique)
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    es.add((L[i], L[j]) if L[i] < L[j] else (L[j], L[i]))
            return es

        def build_solution(order):
            # edges that must be covered
            uncovered = set()
            for v in I:
                for u in I_neigh[v]:
                    uncovered.add((u, v))
            # C internal edges
            for a in range(k):
                for b in range(a + 1, k):
                    uncovered.add((a, b))

            cliques = []
            # group I vertices by neighborhood
            from collections import defaultdict
            by_neigh = defaultdict(list)
            for v in order:
                by_neigh[frozenset(I_neigh[v])].append(v)

            # Build candidate cliques:
            #  - C-only cliques (pairs, triples)
            #  - cliques = {v} + subset of S(v)
            #  - cliques = {v1, v2} + S(v1) where S(v1)==S(v2)
            candidates = []
            # C pairs
            for a in range(k):
                for b in range(a + 1, k):
                    candidates.append([a, b])
            # C triples
            for a in range(k):
                for b in range(a + 1, k):
                    for c in range(b + 1, k):
                        candidates.append([a, b, c])
            # single I vertex with full neighborhood
            for v in I:
                ns = I_neigh[v]
                if ns:
                    candidates.append([v] + list(ns))
                # also partial subsets (small)
            # pair I vertices with same neighborhood
            for ns, vs in by_neigh.items():
                if len(vs) >= 2 and ns:
                    for i in range(len(vs)):
                        for j in range(i + 1, len(vs)):
                            candidates.append([vs[i], vs[j]] + list(ns))

            # precompute edge set per candidate
            cand_edges = [clique_edges(c) for c in candidates]

            # Greedy set cover
            while uncovered and time.time() < deadline:
                best_i = -1
                best_cov = 0
                # iterate candidates, find max coverage among uncovered
                for i, ce in enumerate(cand_edges):
                    if not ce:
                        continue
                    cov = len(ce & uncovered)
                    if cov > best_cov:
                        best_cov = cov
                        best_i = i
                        if best_cov == len(uncovered):
                            break
                if best_i < 0:
                    # should not happen; fallback single edge
                    e = next(iter(uncovered))
                    cliques.append(list(e))
                    uncovered.discard(e)
                    continue
                cliques.append(candidates[best_i])
                uncovered -= cand_edges[best_i]

            return cliques

        # multiple orderings
        import random as _r
        base_order = I[:]
        orders = [base_order]
        # reverse
        orders.append(base_order[::-1])
        # by neighborhood size desc
        orders.append(sorted(I, key=lambda v: (-len(I_neigh[v]), v)))
        orders.append(sorted(I, key=lambda v: (len(I_neigh[v]), v)))
        # a few random shuffles
        rr = _r.Random(seed + 777)
        for _ in range(4):
            o = I[:]
            rr.shuffle(o)
            orders.append(o)

        best = None
        for o in orders:
            if time.time() > deadline:
                break
            sol = build_solution(o)
            if best is None or len(sol) < len(best):
                best = sol
        return best

    start = time.time()
    deadline = start + 120.0  # safety margin
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        n, k, p, seed = spec
        # allocate remaining time, keep margin
        rem = deadline - time.time()
        per = max(1.0, rem / (len(specs) - len(out)))
        local_deadline = min(deadline - 0.5, time.time() + per)
        cliques = _cover_one(n, k, p, seed, local_deadline)
        # validate & sanitize
        cleaned = []
        for c in cliques:
            L = [int(x) for x in c]
            if len(set(L)) >= 2:
                cleaned.append(L)
        out.append(cleaned)
    return out
# EVOLVE-BLOCK-END

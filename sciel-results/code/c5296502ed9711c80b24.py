import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def max_clique(V, adj):
        V = set(V)
        if not V:
            return []
        best = []
        def bk(R, P, X):
            nonlocal best
            if len(R) + len(P) <= len(best):
                return
            if not P and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            if not P:
                return
            # Pivot: choose u from P|X that maximizes |N(u) ∩ P|
            pool = P | X
            u = max(pool, key=lambda x: len(adj.get(x, set()) & P))
            # Candidates: vertices in P that are NOT adjacent to u
            candidates = P - adj.get(u, set())
            for v in list(candidates):
                R.add(v)
                bk(R, P & adj.get(v, set()), X & adj.get(v, set()))
                R.remove(v)
                P.remove(v)
                X.add(v)
                if len(R) + len(P) <= len(best):
                    return
        bk(set(), V, set())
        return best

    def solve(n, k, p, seed):
        if time.time() > deadline:
            return []

        rng = random.Random(seed)
        cross = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if rng.random() < p:
                    nbrs.add(u)
            cross[v] = nbrs

        cliques = []
        covered_KE = set()

        # Phase 1: Cover K-I edges
        for v in range(k, n):
            remaining = set(cross[v])
            while remaining:
                if time.time() > deadline:
                    # Fallback: single vertex cliques
                    for u in remaining:
                        cliques.append([u, v])
                    remaining.clear()
                    break

                avail_adj = {u: set() for u in remaining}
                for u in remaining:
                    for w in remaining:
                        if u == w:
                            continue
                        edge = (min(u, w), max(u, w))
                        if edge not in covered_KE:
                            avail_adj[u].add(w)

                mc = max_clique(remaining, avail_adj)
                if not mc:
                    mc = [next(iter(remaining))]

                clique = sorted(mc) + [v]
                cliques.append(clique)

                mcl = sorted(mc)
                for i in range(len(mcl)):
                    for j in range(i + 1, len(mcl)):
                        covered_KE.add((mcl[i], mcl[j]))

                remaining -= set(mc)

        # Phase 2: Cover remaining K-K edges
        remaining_edges = set()
        for i in range(k):
            for j in range(i + 1, k):
                if (i, j) not in covered_KE:
                    remaining_edges.add((i, j))

        while remaining_edges:
            if time.time() > deadline:
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break

            rem_adj = {u: set() for u in range(k)}
            edge_vertices = set()
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                edge_vertices.add(u)
                edge_vertices.add(v)

            mc = max_clique(edge_vertices, rem_adj)
            if not mc:
                e = next(iter(remaining_edges))
                mc = [e[0], e[1]]

            cl = sorted(mc)
            cliques.append(cl)

            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    e = (min(cl[i], cl[j]), max(cl[i], cl[j]))
                    remaining_edges.discard(e)
                    # weights.pop(e, None) # Not needed if we rebuild or check membership

        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve(*spec)
            results.append(res)

    cleaned = [[[int(x) for x in c] for c in clist] for clist in results]
    return cleaned

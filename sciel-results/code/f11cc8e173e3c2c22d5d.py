"""Rewritten: strict edge-owned clique cover, greedy one-vertex-at-a-time expansion."""
# EVOLVE-BLOCK-START
def construct():
    import random
    from collections import Counter

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _build(n, k, p, seed):
        rng = random.Random(seed)
        cross = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross.append((u, v))
        c_nei = {u: set() for u in range(k)}
        i_nei = {v: set() for v in range(k, n)}
        for u, v in cross:
            c_nei[u].add(v)
            i_nei[v].add(u)

        all_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_edges.add((u, v))
        for u, v in cross:
            all_edges.add((u, v))

        remaining = set(all_edges)
        edge_owner = {}
        cliques = []

        def edge_key(e):
            a, b = e
            is_cross = (a >= k) or (b >= k)
            da = len(i_nei[a]) if a >= k else len(c_nei[a])
            db = len(i_nei[b]) if b >= k else len(c_nei[b])
            return (0 if is_cross else 1, -(da + db), a, b)

        while remaining:
            e = min(remaining, key=edge_key)
            a, b = e
            cur = [a, b]
            # candidate clique vertices (0..k-1) that connect to all of cur and whose
            # edges to cur are still in remaining
            cand = set(range(k))
            for c in cur:
                cand &= (i_nei[c] if c >= k else c_nei[c])
            # order candidates by degree for better packing
            for x in sorted(cand, key=lambda x: -len(c_nei[x])):
                if x in cur:
                    continue
                ok = True
                for c in cur:
                    ee = (min(x, c), max(x, c))
                    if ee not in remaining:
                        ok = False
                        break
                if ok:
                    cur.append(x)
            ci = len(cliques)
            cliques.append(sorted(cur))
            for i in range(len(cur)):
                for j in range(i + 1, len(cur)):
                    ee = (min(cur[i], cur[j]), max(cur[i], cur[j]))
                    if ee in remaining:
                        edge_owner[ee] = ci
                        remaining.discard(ee)
        return cliques

    out = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        cliques = _build(n, k, p, seed)
        out.append([list(map(int, c)) for c in cliques])
    return out
# EVOLVE-BLOCK-END

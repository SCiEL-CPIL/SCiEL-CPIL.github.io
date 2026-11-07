"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(n, k, p, seed):
        """
        Clique partition of a split graph (K=clique[0,k), I=independent[k,n)).
        Triangles are exactly {i,u,v} with i in I, u<v in K (I independent).
        Greedy triangle packing with K-clique fragmentation, then local-search
        refinement (merge K-cliques / re-pack / drop redundant K-cliques).
        """
        rng = Random(seed)
        cross = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            cross[v] = nbrs

        def build_solution(packed, Kcliques):
            cliques = []
            for i, (u, v) in packed.items():
                cliques.append(sorted([i, u, v]))
            for s in Kcliques:
                if len(s) >= 2:
                    cliques.append(sorted(s))
            return cliques

        def tri_edges_of(packed):
            te = set()
            for i, (u, v) in packed.items():
                te.add((min(u, v), max(u, v)))
            return te

        def refine(cliques, packed):
            Kcliques = []
            Triangles = []
            for c in cliques:
                if len(c) == 3:
                    iv = [x for x in c if x >= k]
                    kv = sorted(x for x in c if x < k)
                    if len(iv) == 1 and len(kv) == 2:
                        Triangles.append((iv[0], tuple(kv)))
                    elif all(x < k for x in c):
                        Kcliques.append(set(c))
                else:
                    if all(x < k for x in c):
                        Kcliques.append(set(c))
            changed = False
            te = tri_edges_of(packed)

            def union_cross(s):
                u = set()
                for x in s:
                    u.update(cross.get(x, ()))
                return u

            # merge K-cliques when safe: all cross-neighbors packed AND no internal
            # K-edge is a triangle edge
            i = 0
            while i < len(Kcliques):
                j = i + 1
                while j < len(Kcliques):
                    if Kcliques[i] and Kcliques[j]:
                        merged = Kcliques[i] | Kcliques[j]
                        mu = union_cross(merged)
                        ok = mu and all(x in packed for x in mu)
                        if ok:
                            # no internal triangle edge
                            for a in merged:
                                for b in merged:
                                    if a < b and (a, b) in te:
                                        ok = False
                                        break
                                if not ok:
                                    break
                        if ok:
                            Kcliques[i] = merged
                            del Kcliques[j]
                            changed = True
                            continue
                    j += 1
                i += 1

            Kcliques = [s for s in Kcliques if len(s) >= 2]
            newcliques = build_solution(packed, Kcliques)
            return newcliques, changed

        # ---- initial greedy packing ----
        packed = {}
        freeK = set(range(k))
        Kcliques = [set(range(k))]

        order = sorted(range(k, n), key=lambda i: -len(cross[i]))

        for i in order:
            nbrs = cross[i]
            if len(nbrs) < 2:
                continue
            cand = sorted(u for u in nbrs if u in freeK)
            if len(cand) < 2:
                continue
            best_pair = None
            best_cost = None
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    u, v = cand[a], cand[b]
                    tmp = set(freeK)
                    tmp.discard(u)
                    tmp.discard(v)
                    seen = set()
                    comps = 0
                    for x in tmp:
                        if x in seen:
                            continue
                        comps += 1
                        stack = [x]
                        seen.add(x)
                        while stack:
                            y = stack.pop()
                            for z in range(k):
                                if z in tmp and z not in seen:
                                    seen.add(z)
                                    stack.append(z)
                    cost = comps
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_pair = (u, v)
                    if best_cost == 0:
                        break
                if best_cost == 0:
                    break
            if best_pair is None:
                continue
            u, v = best_pair
            freeK.discard(u)
            freeK.discard(v)
            seen = set()
            newKcliques = []
            for x in sorted(freeK):
                if x in seen:
                    continue
                comp = set()
                stack = [x]
                seen.add(x)
                while stack:
                    y = stack.pop()
                    comp.add(y)
                    for z in range(k):
                        if z in freeK and z not in seen:
                            seen.add(z)
                            stack.append(z)
                newKcliques.append(comp)
            Kcliques = [c for c in newKcliques if len(c) >= 2]
            packed[i] = (u, v)

        cliques = build_solution(packed, Kcliques)

        # ---- refinement loop ----
        for _ in range(8):
            if time.time() > deadline:
                break
            newcliques, changed = refine(cliques, packed)
            if not changed:
                break
            packed = {}
            for c in newcliques:
                if len(c) == 3:
                    iv = [x for x in c if x >= k]
                    kv = sorted(x for x in c if x < k)
                    if len(iv) == 1 and len(kv) == 2:
                        packed[iv[0]] = tuple(kv)
            cliques = newcliques

        # ---- final drop of redundant K-cliques ----
        te = tri_edges_of(packed)
        final = []
        for c in cliques:
            if len(c) == 3:
                final.append(sorted(c))
            else:
                keep = False
                for a in range(len(c)):
                    for b in range(a + 1, len(c)):
                        if (min(c[a], c[b]), max(c[a], c[b])) not in te:
                            keep = True
                            break
                    if keep:
                        break
                if keep and len(c) >= 2:
                    final.append(sorted(c))
        return final

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        if time.time() > deadline:
            break
        cliques = clique_partition(n, k, p, seed)
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
# EVOLVE-BLOCK-END

import time
from random import Random
import numpy as np

def construct():
    deadline = time.time() + 140

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def popcount(x):
        return bin(x).count('1')

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        orig_adj = [0] * n
        for u, v in edges:
            orig_adj[u] |= (1 << v)
            orig_adj[v] |= (1 << u)

        adj = orig_adj[:]
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        remaining = set(edge_set)
        cliques = []

        def is_clique_in_orig(verts):
            vs = list(verts)
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    if not (orig_adj[vs[i]] & (1 << vs[j])):
                        return False
            return True

        def extend_clique(seed_verts):
            clique = list(seed_verts)
            common = (1 << n) - 1
            for v in clique:
                common &= adj[v]
                common &= ~(1 << v)
            for v in clique:
                common &= ~(1 << v)
            while common:
                bv = -1
                bd = -1
                tmp = common
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    d = popcount(adj[v] & common)
                    if d > bd:
                        bd = d
                        bv = v
                    tmp ^= lsb
                if bv < 0:
                    break
                clique.append(bv)
                common &= adj[bv]
                common &= ~(1 << bv)
            return clique

        def remove_clique_edges(clique):
            cset = set(clique)
            for u, v in list(remaining):
                if u in cset and v in cset:
                    remaining.discard((u, v))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        # Build triangles from original graph
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if orig_adj[u] & (1 << v):
                    common = orig_adj[u] & orig_adj[v] & ~((1 << u) | (1 << v))
                    tmp = common
                    while tmp:
                        lsb = tmp & (-tmp)
                        w = lsb.bit_length() - 1
                        if w > v:
                            triangles.append((u, v, w))
                        tmp ^= lsb

        deg_orig = [popcount(orig_adj[v]) for v in range(n)]
        tri_list = []
        for tri in triangles:
            u, v, w = tri
            e1 = (min(u, v), max(u, v))
            e2 = (min(u, w), max(u, w))
            e3 = (min(v, w), max(v, w))
            if e1 in edge_set and e2 in edge_set and e3 in edge_set:
                score = deg_orig[u] + deg_orig[v] + deg_orig[w]
                tri_list.append((score, tri))
        tri_list.sort(key=lambda x: -x[0])

        # Phase 1: Extract triangles (extended into larger cliques)
        for score, tri in tri_list:
            if time.time() > deadline - 30:
                break
            u, v, w = tri
            e1 = (min(u, v), max(u, v))
            e2 = (min(u, w), max(u, w))
            e3 = (min(v, w), max(v, w))
            if e1 not in remaining or e2 not in remaining or e3 not in remaining:
                continue
            clique = extend_clique([u, v, w])
            if len(clique) < 3:
                continue
            if not is_clique_in_orig(clique):
                clique = [u, v, w]
            remove_clique_edges(clique)
            cliques.append(clique)

        # Phase 2: Extract remaining cliques greedily
        while remaining and time.time() < deadline - 20:
            rem_deg = [0] * n
            for u, v in remaining:
                rem_deg[u] += 1
                rem_deg[v] += 1

            best_start = -1
            best_d = -1
            for v in range(n):
                if rem_deg[v] > best_d:
                    best_d = rem_deg[v]
                    best_start = v
            if best_start < 0:
                break

            clique = [best_start]
            common = adj[best_start] & ~(1 << best_start)
            while common:
                bv = -1
                bd = -1
                tmp = common
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    d = popcount(adj[v] & common)
                    if d > bd:
                        bd = d
                        bv = v
                    tmp ^= lsb
                if bv < 0:
                    break
                clique.append(bv)
                common &= adj[bv]
                common &= ~(1 << bv)

            if len(clique) >= 2 and is_clique_in_orig(clique):
                remove_clique_edges(clique)
                cliques.append(clique)
            else:
                e = next(iter(remaining))
                remaining.discard(e)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
                cliques.append(list(e))

        # Leftover edges
        for e in list(remaining):
            cliques.append(list(e))
        remaining.clear()

        # Phase 3: Local search - merge cliques whose union is a clique in original graph
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n_cliques = len(cliques)
            for i in range(n_cliques):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n_cliques):
                    union = set(cliques[i]) | set(cliques[j])
                    if len(union) < 2:
                        continue
                    if is_clique_in_orig(union):
                        cliques[i] = list(union)
                        cliques.pop(j)
                        improved = True
                        break
                if improved:
                    break

        # Final verification
        edge_map = {}
        for ci, clique in enumerate(cliques):
            cl = list(clique)
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    e = (min(cl[a], cl[b]), max(cl[a], cl[b]))
                    if e in edge_set:
                        if e not in edge_map:
                            edge_map[e] = ci

        for e in edge_set:
            if e not in edge_map:
                cliques.append(list(e))

        final_cliques = []
        for clique in cliques:
            cl = list(clique)
            has_edge = False
            for a in range(len(cl)):
                for b in range(a + 1, len(cl)):
                    e = (min(cl[a], cl[b]), max(cl[a], cl[b]))
                    if e in edge_set:
                        has_edge = True
                        break
                if has_edge:
                    break
            if has_edge:
                final_cliques.append(clique)

        return final_cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result

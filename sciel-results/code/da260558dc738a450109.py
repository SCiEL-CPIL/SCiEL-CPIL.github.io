import time
import random

def construct():
    deadline = time.time() + 145.0
    n = 40

    def creates_c6(adj, u, v):
        nu = adj[u]
        nv = adj[v]
        for x in nu:
            if x == v:
                continue
            nx = adj[x]
            for z in nv:
                if z == u or z == x:
                    continue
                nz = adj[z]
                for y in (nx & nz):
                    if y != u and y != v and y != x and y != z:
                        return True
        return False

    def find_c6(adj):
        for a in range(n):
            na = adj[a]
            for b in na:
                if b <= a:
                    continue
                for x in na:
                    if x == b:
                        continue
                    nx = adj[x]
                    for z in adj[b]:
                        if z == a or z == x:
                            continue
                        nz = adj[z]
                        common = nx & nz
                        for y in common:
                            if y != a and y != b and y != x and y != z:
                                return [a, x, y, z, b]
        return None

    def make_adj(edges):
        adj = [set() for _ in range(n)]
        es = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            es.add((min(u, v), max(u, v)))
        return adj, es

    def build_once(seed):
        r = random.Random(seed)
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u in range(n):
            for v in range(u + 1, n):
                if r.random() < 0.55:
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
        while time.time() < deadline - 5:
            cyc = find_c6(adj)
            if cyc is None:
                break
            a, x, y, z, b = cyc
            ce = [(a, x), (x, y), (y, z), (z, b), (b, a)]
            ue, ve = r.choice(ce)
            adj[ue].discard(ve)
            adj[ve].discard(ue)
            edge_set.discard((min(ue, ve), max(ue, ve)))
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        r.shuffle(all_edges)
        for _ in range(3):
            for u, v in all_edges:
                if time.time() > deadline - 5:
                    break
                if (u, v) in edge_set:
                    continue
                if not creates_c6(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
        return [(u, v) for (u, v) in edge_set]

    def improve_from(edges, seed):
        r = random.Random(seed)
        adj, edge_set = make_adj(edges)
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        r.shuffle(all_edges)
        for _ in range(3):
            for u, v in all_edges:
                if time.time() > deadline - 5:
                    break
                if (u, v) in edge_set:
                    continue
                if not creates_c6(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
        return [(u, v) for (u, v) in edge_set]

    base = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                base.append([5 * k + a, 5 * k + b])
    best_edges = [list(e) for e in base]
    best_count = len(base)

    imp = improve_from(best_edges, 999)
    if len(imp) > best_count:
        best_count = len(imp)
        best_edges = [list(e) for e in imp]

    it = 0
    while time.time() < deadline - 3:
        it += 1
        edges = build_once(1000 + it)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
            imp = improve_from(best_edges, 2000 + it)
            if len(imp) > best_count:
                best_count = len(imp)
                best_edges = [list(e) for e in imp]

    return [list(e) for e in best_edges]

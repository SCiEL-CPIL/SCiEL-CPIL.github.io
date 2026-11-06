import time
import random

def construct():
    start = time.time()
    DEADLINE = 138.0

    def now():
        return time.time()

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges.add((u, v))
        return edges, adj_mask

    def is_clique_mask(m, adj_mask):
        t = m
        while t:
            lsb = t & (-t)
            v = lsb.bit_length() - 1
            if (m & ~lsb) & ~adj_mask[v]:
                return False
            t ^= lsb
        return True

    def mask_to_list(m):
        out = []
        while m:
            lsb = m & (-m)
            out.append(lsb.bit_length() - 1)
            m ^= lsb
        return out

    def edge_of(a, b):
        return (a, b) if a < b else (b, a)

    def bron_kerbosch_maximal(adj_mask, n, time_limit, cap=60000):
        full = (1 << n) - 1
        out = []
        def bk(R, P, X):
            if now() > time_limit:
                return
            if len(out) >= cap:
                return
            if P == 0:
                if X == 0 and R:
                    out.append(R)
                return
            PX = P | X
            best_u = -1
            best_c = -1
            m = PX
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                c = (P & adj_mask[u]).bit_count()
                if c > best_c:
                    best_c = c
                    best_u = u
                m ^= lsb
            cand = P & ~adj_mask[best_u]
            m = cand
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                bk(R | lsb, P & adj_mask[v], X & adj_mask[v])
                P &= ~lsb
                X |= lsb
        bk(0, full, 0)
        return out

    def valid_partition(cliques, edges, adj_mask):
        covered = set()
        for c in cliques:
            if len(c) < 2:
                return False
            m = 0
            for v in c:
                m |= 1 << v
            if not is_clique_mask(m, adj_mask):
                return False
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    e = edge_of(c[i], c[j])
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == edges

    def greedy_partition(edges, adj_mask, n, deadline, rng):
        remaining = set(edges)
        if not remaining:
            return []
        max_cliques = bron_kerbosch_maximal(adj_mask, n, deadline)
        cands = []
        for m in max_cliques:
            k = m.bit_count()
            if k >= 3:
                cands.append((m, k))
        cands.sort(key=lambda x: x[1], reverse=True)
        if len(cands) > 60000:
            cands = cands[:60000]
        c_edges = []
        for m, k in cands:
            vs = mask_to_list(m)
            es = set()
            for i in range(k):
                for j in range(i + 1, k):
                    es.add(edge_of(vs[i], vs[j]))
            c_edges.append(es)
        edge_to_cands = {}
        for idx, es in enumerate(c_edges):
            for e in es:
                edge_to_cands.setdefault(e, []).append(idx)
        avail = [len(es) for es in c_edges]
        sizes = [len(es) for es in c_edges]
        used = [False] * len(cands)
        cliques = []
        while remaining and now() < deadline:
            best = -1
            best_size = -1
            for idx in range(len(cands)):
                if used[idx]:
                    continue
                if avail[idx] != sizes[idx]:
                    continue
                s = sizes[idx]
                if s > best_size:
                    best_size = s
                    best = idx
            if best == -1:
                e = rng.choice(list(remaining))
                cliques.append(sorted([e[0], e[1]]))
                remaining.discard(e)
                continue
            m = cands[best][0]
            cliques.append(sorted(mask_to_list(m)))
            used[best] = True
            for e in c_edges[best]:
                remaining.discard(e)
                for idx in edge_to_cands[e]:
                    if not used[idx]:
                        avail[idx] -= 1
        for e in list(remaining):
            cliques.append(sorted([e[0], e[1]]))
        return cliques

    def merge_local_search(cliques, adj_mask, n, deadline):
        masks = []
        for c in cliques:
            m = 0
            for v in c:
                m |= 1 << v
            masks.append(m)
        changed = True
        while changed and now() < deadline:
            changed = False
            i = 0
            while i < len(masks) and now() < deadline:
                a = masks[i]
                j = i + 1
                merged = False
                while j < len(masks):
                    b = masks[j]
                    u = a | b
                    if is_clique_mask(u, adj_mask):
                        masks[i] = u
                        del masks[j]
                        changed = True
                        merged = True
                        break
                    j += 1
                if not merged:
                    i += 1
        return [sorted(mask_to_list(m)) for m in masks]

    def branch_and_bound(edges, adj_mask, n, deadline, initial_best):
        max_cliques = bron_kerbosch_maximal(adj_mask, n, deadline, cap=40000)
        edge_cliques = {}
        clique_edge_set = {}
        for m in max_cliques:
            vs = mask_to_list(m)
            es = set()
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    es.add(edge_of(vs[i], vs[j]))
            clique_edge_set[m] = es
            for e in es:
                edge_cliques.setdefault(e, []).append(m)
        W = max((m.bit_count() for m in max_cliques), default=1)
        Wmax_edges = W * (W - 1) // 2
        best = [initial_best if initial_best is not None else 10 ** 9]
        best_sol = [None]
        nodes = [0]

        def dfs(remaining, count):
            if now() > deadline:
                return
            nodes[0] += 1
            if nodes[0] > 400000:
                return
            if not remaining:
                if count < best[0]:
                    best[0] = count
                    best_sol[0] = list(current)
                return
            rem = len(remaining)
            lb1 = (rem + Wmax_edges - 1) // Wmax_edges
            rd = [0] * n
            for (u, v) in remaining:
                rd[u] += 1
                rd[v] += 1
            lb2 = 0
            if W > 1:
                for v in range(n):
                    if rd[v] > 0:
                        b = (rd[v] + (W - 1) - 1) // (W - 1)
                        if b > lb2:
                            lb2 = b
            lb = lb1 if lb1 > lb2 else lb2
            if count + lb >= best[0]:
                return
            # pick max remaining-degree edge
            best_e = None
            best_d = -1
            for (u, v) in remaining:
                d = rd[u] + rd[v]
                if d > best_d:
                    best_d = d
                    best_e = (u, v)
            u, v = best_e
            cands = [m for m in edge_cliques.get(best_e, []) if clique_edge_set[m] <= remaining]
            if not cands:
                cands = [(1 << u) | (1 << v)]
            cands.sort(key=lambda m: m.bit_count(), reverse=True)
            for m in cands:
                if now() > deadline:
                    return
                ces = clique_edge_set.get(m, {(u, v)})
                new_rem = remaining - ces
                if count + 1 + ((len(new_rem) + Wmax_edges - 1) // Wmax_edges) >= best[0]:
                    continue
                current.append(m)
                dfs(new_rem, count + 1)
                current.pop()

        current = []
        dfs(set(edges), 0)
        if best_sol[0] is not None:
            return [sorted(mask_to_list(m)) for m in best_sol[0]], best[0]
        return None, best[0]

    def solve_instance(n, p, seed, budget):
        edges, adj_mask = random_graph(n, p, seed)
        end = now() + budget
        best_cliques = None
        best_count = 10 ** 9
        if n <= 40 and now() < end - 15:
            sol, cnt = branch_and_bound(edges, adj_mask, n, min(end - 5, now() + 25), None)
            if sol is not None and valid_partition(sol, edges, adj_mask):
                best_cliques = sol
                best_count = cnt
        restart = 0
        while now() < end - 2:
            rng = random.Random(seed * 7919 + restart * 131)
            cliques = greedy_partition(edges, adj_mask, n, end - 2, rng)
            cliques = merge_local_search(cliques, adj_mask, n, end - 2)
            if valid_partition(cliques, edges, adj_mask):
                if len(cliques) < best_count:
                    best_count = len(cliques)
                    best_cliques = cliques
            restart += 1
        if best_cliques is None:
            best_cliques = [list(e) for e in edges]
        return best_cliques

    t1 = now()
    b1 = min(45, DEADLINE - (t1 - start) - 90)
    if b1 < 12:
        b1 = 12
    cliques1 = solve_instance(32, 0.6, 601, b1)

    t2 = now()
    b2 = DEADLINE - (t2 - start) - 3
    if b2 < 12:
        b2 = 12
    cliques2 = solve_instance(64, 0.55, 602, b2)

    return [cliques1, cliques2]

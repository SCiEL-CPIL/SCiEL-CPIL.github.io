import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 55.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    def time_up():
        return time.time() - t0 > DEADLINE - 2.5

    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    best = fallback()
    best_size = 2
    rng = random.Random(20240607)

    # ---- exact verifier: max-flow with vertex splitting -------------------
    # Non-branch vertices capacity 1 (each usable by at most one path);
    # branch vertices unlimited. A flow of 1 per pair => internally disjoint
    # (vertex) paths. We route pairs one at a time on the residual graph.
    def max_flow_paths(bs, max_attempts=6):
        bs = set(bs)
        sb = sorted(bs)
        pairs = list(combinations(sb, 2))
        m = len(pairs)
        if m == 0:
            return []

        def vin(v): return 2 * v
        def vout(v): return 2 * v + 1
        def to_vertex(node): return node // 2
        def is_in(node): return node % 2 == 0

        total_nodes = 2 * n

        def build_network():
            flow_adj = [[] for _ in range(total_nodes)]
            edge_info = []
            def add_edge(u, v, cap):
                idx = len(edge_info)
                edge_info.append([u, v, cap])
                flow_adj[u].append((v, idx))
                edge_info.append([v, u, 0])
                flow_adj[v].append((u, idx + 1))
            for v in range(n):
                add_edge(vin(v), vout(v), float('inf') if v in bs else 1)
            for a, b in edges:
                add_edge(vout(a), vin(b), 1)
                add_edge(vout(b), vin(a), 1)
            return flow_adj, edge_info

        def bfs_residual(flow_adj, edge_info, s, t):
            parent_edge = {s: None}
            q = deque([s])
            while q:
                curr = q.popleft()
                if curr == t:
                    break
                for nb, eidx in flow_adj[curr]:
                    if edge_info[eidx][2] > 0 and nb not in parent_edge:
                        parent_edge[nb] = eidx
                        if nb == t:
                            break
                        q.append(nb)
            if t not in parent_edge:
                return None
            path_edges = []
            node = t
            while node != s:
                eidx = parent_edge[node]
                path_edges.append(eidx)
                node = edge_info[eidx][0]
            path_edges.reverse()
            return path_edges

        # orderings of pairs to try
        def tightness(p):
            u, v = p
            return len((adj[u] & adj[v]) - bs)
        base = list(pairs)
        orderings = [
            sorted(base, key=tightness),
            sorted(base, key=lambda p: -tightness(p)),
            base,
        ]
        for _ in range(3):
            sh = list(base)
            rng.shuffle(sh)
            orderings.append(sh)

        for ordering in orderings[:max_attempts]:
            flow_adj, edge_info = build_network()
            paths_found = []
            ok = True
            for u, v in ordering:
                pe = bfs_residual(flow_adj, edge_info, vout(u), vin(v))
                if pe is None:
                    ok = False
                    break
                for eidx in pe:
                    edge_info[eidx][2] -= 1
                    edge_info[eidx ^ 1][2] += 1
                vp = [u]
                node = vout(u)
                for eidx in pe:
                    nb = edge_info[eidx][1]
                    if is_in(nb):
                        w = to_vertex(nb)
                        if w != vp[-1]:
                            vp.append(w)
                    node = nb
                if vp[-1] != v:
                    vp.append(v)
                internal = set(vp[1:-1])
                if internal & bs:
                    ok = False
                    break
                for i in range(len(vp) - 1):
                    if vp[i + 1] not in adj[vp[i]]:
                        ok = False
                        break
                if not ok:
                    break
                paths_found.append(vp)
            if ok:
                return paths_found
        return None

    # ---- cheap pre-filter: directed-edge-disjointness --------------------
    # Necessary condition: assign each pair a distinct directed edge.
    # (Every path of a pair consumes >=1 directed edge; edges are shared by
    #  at most one path, so the pairs need a matching into directed edges.)
    def edge_disjoint_precheck(bs):
        sb = sorted(bs)
        pairs = list(combinations(sb, 2))
        # for each pair, the set of directed edges (u->x, x->u) it could use
        # We just need a matching: try to assign distinct directed edges.
        # Build adjacency: pair -> list of directed-edge ids
        ded = {}
        eid = 0
        for a, b in edges:
            ded[(a, b)] = eid; eid += 1
            ded[(b, a)] = eid; eid += 1
        pair_edges = []
        for u, v in pairs:
            opts = []
            for x in adj[u]:
                opts.append(ded[(u, x)])
            for x in adj[v]:
                opts.append(ded[(x, v)])
            pair_edges.append(set(opts))
        # greedy matching by increasing option count
        order = sorted(range(len(pairs)), key=lambda i: len(pair_edges[i]))
        used = set()
        for i in order:
            if not pair_edges[i]:
                return False
            chosen = None
            for e in pair_edges[i]:
                if e not in used:
                    chosen = e
                    break
            if chosen is None:
                return False
            used.add(chosen)
        return True

    def try_set(bs):
        nonlocal best, best_size
        bs = sorted(bs)
        if len(bs) <= best_size:
            return False
        if not edge_disjoint_precheck(bs):
            return False
        paths = max_flow_paths(bs)
        if paths is not None:
            best = {'branches': [int(x) for x in bs],
                    'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(bs)
            return True
        return False

    # ---- candidate generation --------------------------------------------
    # 1) degree-greedy construction (dense core first)
    def greedy_build():
        cur = []
        pool = list(range(n))
        pool.sort(key=lambda v: len(adj[v]), reverse=True)
        if pool:
            cur = [pool[0]]
        while True:
            if time_up():
                break
            best_v = None
            best_score = -1
            for v in pool:
                if v in cur:
                    continue
                sc = sum(1 for b in cur if v in adj[b])
                sc += 0.001 * len(adj[v])
                if sc > best_score:
                    best_score = sc
                    best_v = v
            if best_v is None:
                break
            newset = cur + [best_v]
            if edge_disjoint_precheck(newset):
                cur = newset
                if len(cur) > best_size:
                    if try_set(cur):
                        pass
            else:
                # cannot keep growing with this vertex; skip it
                pool.remove(best_v)
                if not pool:
                    break
                continue
            if len(cur) == n:
                break
        return cur

    # 2) Bron-Kerbosch maximal cliques
    def bron_kerbosch(R, Pset, X, result, tlimit):
        if time.time() > tlimit:
            return
        if not Pset and not X:
            if len(R) > best_size:
                result.append(list(R))
            return
        cands = Pset | X
        if cands:
            pivot = max(cands, key=lambda v: len(Pset & adj[v]))
            to_try = Pset - adj[pivot]
        else:
            to_try = Pset
        for v in list(to_try):
            bron_kerbosch(R | {v}, Pset & adj[v], X & adj[v], result, tlimit)
            Pset.discard(v)
            X.add(v)

    candidates = []

    # greedy build
    g = greedy_build()
    if g:
        candidates.append(g)

    # maximal cliques
    cliques = []
    bron_kerbosch(set(), set(range(n)), set(), cliques, time.time() + 8.0)
    cliques.sort(key=len, reverse=True)
    for c in cliques:
        candidates.append(c)

    # random dense subsets of several sizes
    deg_sorted = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    seen = set()
    for k in range(3, min(n + 1, 16)):
        for _ in range(40):
            if time_up():
                break
            sub = tuple(sorted(rng.sample(deg_sorted[:min(20, n)], k)))
            if sub not in seen:
                seen.add(sub)
                candidates.append(list(sub))

    # greedy extension from the current best
    if not time_up():
        cur = list(best['branches'])
        for _ in range(12):
            if time_up():
                break
            improved = False
            def score(v):
                return sum(1 for b in cur if v in adj[b]) + 0.001 * len(adj[v])
            cands = [v for v in range(n) if v not in cur]
            cands.sort(key=score, reverse=True)
            for v in cands:
                newset = sorted(cur + [v])
                if edge_disjoint_precheck(newset):
                    if try_set(newset):
                        cur = newset
                        improved = True
                        break
            if not improved:
                break

    # evaluate all generated candidates (largest first)
    candidates.sort(key=len, reverse=True)
    seen2 = set()
    for c in candidates:
        if time_up():
            break
        c = sorted(c)
        key = tuple(c)
        if key in seen2:
            continue
        seen2.add(key)
        if len(c) <= best_size:
            continue
        if not edge_disjoint_precheck(c):
            continue
        paths = max_flow_paths(c)
        if paths is not None:
            best = {'branches': [int(x) for x in c],
                    'paths': [[int(x) for x in p] for p in paths]}
            best_size = len(c)

    # final safety
    if time.time() - t0 > DEADLINE - 1.0:
        return best

    return best

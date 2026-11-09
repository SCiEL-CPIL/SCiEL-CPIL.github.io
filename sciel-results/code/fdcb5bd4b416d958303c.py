import time
import random

N = 15
DEADLINE_MARGIN = 50.0  # finish well under the 60s contract limit


def _build_adj(edge_set):
    adj = [set() for _ in range(N)]
    for (a, b) in edge_set:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _deg(adj):
    return [len(adj[i]) for i in range(N)]


def _viol_bad(adj, deg):
    """Return (num_violations, {edge: count_of_violations_it_participates_in})."""
    bad = {}
    nviol = 0
    for u in range(N):
        au = adj[u]
        for v in au:
            if v <= u:
                continue
            av = adj[v]
            for s in au:
                if s == v:
                    continue
                ds = deg[s]
                for t in av:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        nviol += 1
                        for e in ((s, u), (u, v), (v, t)):
                            x, y = e
                            if x > y:
                                x, y = y, x
                            k = (x, y)
                            bad[k] = bad.get(k, 0) + 1
    return nviol, bad


def _valid(adj, deg):
    for u in range(N):
        au = adj[u]
        for v in au:
            if v <= u:
                continue
            av = adj[v]
            for s in au:
                if s == v:
                    continue
                ds = deg[s]
                for t in av:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True


def _greedy_remove(edge_set, deadline):
    """Remove the most-violation-involved edge until no violations remain."""
    adj = _build_adj(edge_set)
    deg = _deg(adj)
    while time.time() < deadline:
        nviol, bad = _viol_bad(adj, deg)
        if nviol == 0:
            break
        worst = max(bad.values())
        cand = [k for k, v in bad.items() if v == worst]
        a, b = cand[0]
        adj[a].discard(b)
        adj[b].discard(a)
        deg[a] -= 1
        deg[b] -= 1
        edge_set.discard((a, b))
    return edge_set


def _add_back(edge_set, deadline, max_passes=40):
    """Local search: try adding each missing edge, keep it if graph stays valid."""
    adj = _build_adj(edge_set)
    deg = _deg(adj)
    for _ in range(max_passes):
        if time.time() >= deadline:
            break
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if (u, v) in edge_set:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                if _valid(adj, deg):
                    edge_set.add((u, v))
                    improved = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
                    deg[u] -= 1
                    deg[v] -= 1
        if not improved:
            break
    return edge_set


def _star():
    return set((0, v) for v in range(1, N))


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_MARGIN

    best = set()

    strategies = []
    # Complete graph (densest start)
    strategies.append(set((a, b) for a in range(N) for b in range(a + 1, N)))
    # A few random dense starts
    rng = random.Random(12345)
    for p in (0.9, 0.8, 0.7):
        es = set()
        for a in range(N):
            for b in range(a + 1, N):
                if rng.random() < p:
                    es.add((a, b))
        strategies.append(es)

    for es in strategies:
        if time.time() >= deadline:
            break
        es = _greedy_remove(set(es), deadline)
        es = _add_back(es, deadline)
        if len(es) > len(best):
            best = set(es)

    if not best:
        best = _star()

    out = []
    for (a, b) in sorted(best):
        out.append([int(a), int(b)])
    return out

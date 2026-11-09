import time
import random
import math
import numpy as np

N = 16
M = 65
ALL_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N)]  # 120
DEADLINE = 140.0  # safety margin under 150s


def _build_adj(edge_set):
    adj = np.zeros((N, N), dtype=np.int8)
    for u, v in edge_set:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj


def _deg(adj):
    return adj.sum(axis=1)


def _tri_eval(adj, deg):
    """Return (max_sum, total_sum, worst_triangle) over all triangles."""
    mx = -1
    tot = 0
    worst = None
    for u in range(N):
        au = adj[u]
        du = int(deg[u])
        for v in range(u + 1, N):
            if au[v]:
                common = np.nonzero(au & adj[v])[0]
                dv = int(deg[v])
                for w in common:
                    if w > v:
                        s = du + dv + int(deg[w])
                        tot += s
                        if s > mx:
                            mx = s
                            worst = (u, v, w)
    if mx < 0:
        return (0, 0, None)
    return (mx, tot, worst)


def _energy(mx, tot):
    return mx + 0.05 * tot


def _base_k88():
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((u, v))
    edges.add((0, 1))
    return edges


def _random_graph():
    return set(random.sample(ALL_PAIRS, M))


def _swap_once(adj, edge_set, e1, e2):
    u1, v1 = e1
    u2, v2 = e2
    adj[u1, v1] = 0
    adj[v1, u1] = 0
    adj[u2, v2] = 1
    adj[v2, u2] = 1
    edge_set.discard(e1)
    edge_set.add(e2)


def _optimize(adj, edge_set, time_limit):
    deg = _deg(adj)
    mx, tot, worst = _tri_eval(adj, deg)
    best_adj = adj.copy()
    best_set = set(edge_set)
    best_mx, best_tot = mx, tot

    T = 2.0
    T_end = 0.02
    start = time.time()
    steps = 0
    while time.time() - start < time_limit:
        steps += 1
        e_cur = _energy(mx, tot)
        # choose move
        use_directed = (worst is not None and random.random() < 0.5)
        if use_directed:
            # break the worst triangle: remove one of its 3 edges
            u, v, w = worst
            cand_edges = [(u, v), (u, w), (v, w)]
            e1 = random.choice(cand_edges)
        else:
            e1 = random.choice(tuple(edge_set))
        # pick a non-edge to add
        nonedges = [p for p in ALL_PAIRS if p not in edge_set]
        e2 = random.choice(nonedges)
        if e1 == e2:
            continue
        # apply
        _swap_once(adj, edge_set, e1, e2)
        deg = _deg(adj)
        nmx, ntot, nworst = _tri_eval(adj, deg)
        e_new = _energy(nmx, ntot)
        if e_new <= e_cur or random.random() < math.exp(-(e_new - e_cur) / max(T, 1e-6)):
            mx, tot, worst = nmx, ntot, nworst
            if (nmx, ntot) < (best_mx, best_tot):
                best_mx, best_tot = nmx, ntot
                best_adj = adj.copy()
                best_set = set(edge_set)
        else:
            # revert
            _swap_once(adj, edge_set, e2, e1)
        # cool
        T = max(T_end, T * 0.9995)
    return best_adj, best_set, best_mx, best_tot, steps


def _polish(adj, edge_set, time_limit):
    deg = _deg(adj)
    mx, tot, _ = _tri_eval(adj, deg)
    improved = True
    start = time.time()
    while improved and time.time() - start < time_limit:
        improved = False
        nonedges = [p for p in ALL_PAIRS if p not in edge_set]
        cur = (mx, tot)
        best_move = None
        best_val = cur
        for e1 in list(edge_set):
            for e2 in nonedges:
                if e1 == e2:
                    continue
                _swap_once(adj, edge_set, e1, e2)
                d = _deg(adj)
                nmx, ntot, _ = _tri_eval(adj, d)
                if (nmx, ntot) < best_val:
                    best_val = (nmx, ntot)
                    best_move = (e1, e2)
                _swap_once(adj, edge_set, e2, e1)
        if best_move is not None:
            e1, e2 = best_move
            _swap_once(adj, edge_set, e1, e2)
            mx, tot = best_val
            improved = True
    return adj, edge_set, mx, tot


def construct():
    t0 = time.time()
    budget = DEADLINE - (time.time() - t0)

    global_best_adj = None
    global_best_set = None
    global_best = (10**9, 10**9)

    seeds = [_base_k88()] + [_random_graph() for _ in range(6)]
    random.shuffle(seeds)

    # Phase 1: SA from multiple seeds
    remaining = lambda: DEADLINE - (time.time() - t0)
    for i, seed in enumerate(seeds):
        if remaining() < 15:
            break
        # allocate time per seed, more for early seeds
        frac = remaining()
        alloc = max(6.0, frac * (0.5 if i == 0 else 0.3))
        adj = _build_adj(seed)
        es = set(seed)
        b_adj, b_set, b_mx, b_tot, _ = _optimize(adj, es, alloc)
        if (b_mx, b_tot) < global_best:
            global_best = (b_mx, b_tot)
            global_best_adj = b_adj
            global_best_set = b_set

    # Phase 2: polish the best found
    if remaining() > 4 and global_best_adj is not None:
        adj = global_best_adj.copy()
        es = set(global_best_set)
        p_adj, p_set, p_mx, p_tot = _polish(adj, es, max(1.0, remaining() - 2))
        if (p_mx, p_tot) < global_best:
            global_best = (p_mx, p_tot)
            global_best_adj = p_adj
            global_best_set = p_set

    # Safety: ensure we have a valid graph
    if global_best_set is None or len(global_best_set) != M:
        global_best_set = _base_k88()
        global_best_adj = _build_adj(global_best_set)

    edges = sorted(global_best_set)
    return {'n': N, 'edges': [[u, v] for u, v in edges]}
